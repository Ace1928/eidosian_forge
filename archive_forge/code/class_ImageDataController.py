import os
from cursive import exception as cursive_exception
import glance_store
from glance_store import backend
from glance_store import location
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
import webob.exc
import glance.api.policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import trust_auth
from glance.common import utils
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _, _LE, _LI
import glance.notifier
from glance.quota import keystone as ks_quota
class ImageDataController(object):

    def __init__(self, db_api=None, store_api=None, policy_enforcer=None, notifier=None):
        db_api = db_api or glance.db.get_api()
        store_api = store_api or glance_store
        notifier = notifier or glance.notifier.Notifier()
        self.policy = policy_enforcer or glance.api.policy.Enforcer()
        self.gateway = glance.gateway.Gateway(db_api, store_api, notifier, self.policy)

    def _restore(self, image_repo, image):
        """
        Restore the image to queued status.

        :param image_repo: The instance of ImageRepo
        :param image: The image will be restored
        """
        try:
            if image_repo and image:
                image.status = 'queued'
                image_repo.save(image)
        except Exception as e:
            msg = _LE('Unable to restore image %(image_id)s: %(e)s') % {'image_id': image.image_id, 'e': encodeutils.exception_to_unicode(e)}
            LOG.exception(msg)

    def _unstage(self, image_repo, image, staging_store):
        """
        Restore the image to queued status and remove data from staging.

        :param image_repo: The instance of ImageRepo
        :param image: The image will be restored
        :param staging_store: The store used for staging
        """
        if CONF.enabled_backends:
            file_path = '%s/%s' % (getattr(CONF, 'os_glance_staging_store').filesystem_store_datadir, image.image_id)
            try:
                loc = location.get_location_from_uri_and_backend(file_path, 'os_glance_staging_store')
                staging_store.delete(loc)
            except (glance_store.exceptions.NotFound, glance_store.exceptions.UnknownScheme):
                pass
        else:
            file_path = str(CONF.node_staging_uri + '/' + image.image_id)[7:]
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except OSError as e:
                    LOG.error(_('Cannot delete staged image data %(fn)s [Errno %(en)d]'), {'fn': file_path, 'en': e.errno})
            else:
                LOG.warning(_('Staged image data not found at %(fn)s'), {'fn': file_path})
        self._restore(image_repo, image)

    @utils.mutating
    def upload(self, req, image_id, data, size):
        try:
            ks_quota.enforce_image_size_total(req.context, req.context.owner)
        except exception.LimitExceeded as e:
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=str(e), request=req)
        backend = None
        if CONF.enabled_backends:
            backend = req.headers.get('x-image-meta-store', CONF.glance_store.default_backend)
            try:
                glance_store.get_store_from_store_identifier(backend)
            except glance_store.UnknownScheme as exc:
                raise webob.exc.HTTPBadRequest(explanation=exc.msg, request=req, content_type='text/plain')
        image_repo = self.gateway.get_repo(req.context)
        image = None
        refresher = None
        cxt = req.context
        try:
            image = image_repo.get(image_id)
            api_pol = api_policy.ImageAPIPolicy(req.context, image, self.policy)
            api_pol.upload_image()
            image.status = 'saving'
            try:
                try:
                    user_plugin = req.environ.get('keystone.token_auth')
                    roles = []
                    for role_info in req.environ.get('keystone.token_info')['token']['roles']:
                        roles.append(role_info['name'])
                    refresher = trust_auth.TokenRefresher(user_plugin, cxt.project_id, roles)
                except Exception as e:
                    LOG.info(_LI('Unable to create trust: %s Use the existing user token.'), encodeutils.exception_to_unicode(e))
                image_repo.save(image, from_state='queued')
                ks_quota.enforce_image_count_uploading(req.context, req.context.owner)
                image.set_data(data, size, backend=backend)
                try:
                    image_repo.save(image, from_state='saving')
                except exception.NotAuthenticated:
                    if refresher is not None:
                        cxt.auth_token = refresher.refresh_token()
                        image_repo.save(image, from_state='saving')
                    else:
                        raise
                try:
                    if refresher is not None:
                        refresher.release_resources()
                except Exception as e:
                    LOG.info(_LI('Unable to delete trust %(trust)s: %(msg)s'), {'trust': refresher.trust_id, 'msg': encodeutils.exception_to_unicode(e)})
            except (glance_store.NotFound, exception.ImageNotFound, exception.Conflict):
                msg = _('Image %s could not be found after upload. The image may have been deleted during the upload, cleaning up the chunks uploaded.') % image_id
                LOG.warning(msg)
                try:
                    image.delete()
                except exception.ImageNotFound:
                    pass
                raise webob.exc.HTTPGone(explanation=msg, request=req, content_type='text/plain')
            except exception.NotAuthenticated:
                msg = _('Authentication error - the token may have expired during file upload. Deleting image data for %s.') % image_id
                LOG.debug(msg)
                try:
                    image.delete()
                except exception.NotAuthenticated:
                    pass
                raise webob.exc.HTTPUnauthorized(explanation=msg, request=req, content_type='text/plain')
        except ValueError as e:
            LOG.debug('Cannot save data for image %(id)s: %(e)s', {'id': image_id, 'e': encodeutils.exception_to_unicode(e)})
            self._restore(image_repo, image)
            raise webob.exc.HTTPBadRequest(explanation=encodeutils.exception_to_unicode(e))
        except glance_store.StoreAddDisabled:
            msg = _('Error in store configuration. Adding images to store is disabled.')
            LOG.exception(msg)
            self._restore(image_repo, image)
            raise webob.exc.HTTPGone(explanation=msg, request=req, content_type='text/plain')
        except exception.InvalidImageStatusTransition as e:
            msg = encodeutils.exception_to_unicode(e)
            LOG.exception(msg)
            raise webob.exc.HTTPConflict(explanation=e.msg, request=req)
        except exception.Forbidden:
            msg = 'Not allowed to upload image data for image %s' % image_id
            LOG.debug(msg)
            raise webob.exc.HTTPForbidden(explanation=msg, request=req)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except glance_store.StorageFull as e:
            msg = _('Image storage media is full: %s') % encodeutils.exception_to_unicode(e)
            LOG.error(msg)
            self._restore(image_repo, image)
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=msg, request=req)
        except exception.StorageQuotaFull as e:
            msg = _('Image exceeds the storage quota: %s') % encodeutils.exception_to_unicode(e)
            LOG.error(msg)
            self._restore(image_repo, image)
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=msg, request=req)
        except exception.ImageSizeLimitExceeded as e:
            msg = _('The incoming image is too large: %s') % encodeutils.exception_to_unicode(e)
            LOG.error(msg)
            self._restore(image_repo, image)
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=msg, request=req)
        except exception.LimitExceeded as e:
            LOG.error(str(e))
            self._restore(image_repo, image)
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=str(e), request=req)
        except glance_store.StorageWriteDenied as e:
            msg = _('Insufficient permissions on image storage media: %s') % encodeutils.exception_to_unicode(e)
            LOG.error(msg)
            self._restore(image_repo, image)
            raise webob.exc.HTTPServiceUnavailable(explanation=msg, request=req)
        except cursive_exception.SignatureVerificationError as e:
            msg = _LE('Signature verification failed for image %(id)s: %(e)s') % {'id': image_id, 'e': encodeutils.exception_to_unicode(e)}
            LOG.error(msg)
            self._restore(image_repo, image)
            raise webob.exc.HTTPBadRequest(explanation=msg)
        except webob.exc.HTTPGone:
            with excutils.save_and_reraise_exception():
                LOG.error(_LE('Failed to upload image data due to HTTP error'))
        except webob.exc.HTTPError:
            with excutils.save_and_reraise_exception():
                LOG.error(_LE('Failed to upload image data due to HTTP error'))
                self._restore(image_repo, image)
        except Exception:
            with excutils.save_and_reraise_exception():
                LOG.error(_LE('Failed to upload image data due to internal error'))
                self._restore(image_repo, image)

    @utils.mutating
    def stage(self, req, image_id, data, size):
        try:
            ks_quota.enforce_image_staging_total(req.context, req.context.owner)
        except exception.LimitExceeded as e:
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=str(e), request=req)
        image_repo = self.gateway.get_repo(req.context)
        try:
            image = image_repo.get(image_id)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        api_pol = api_policy.ImageAPIPolicy(req.context, image, enforcer=self.policy)
        try:
            api_pol.modify_image()
        except exception.Forbidden as e:
            raise webob.exc.HTTPForbidden(explanation=e.msg)

        def _build_staging_store():
            conf = cfg.ConfigOpts()
            try:
                backend.register_opts(conf)
            except cfg.DuplicateOptError:
                pass
            conf.set_override('filesystem_store_datadir', CONF.node_staging_uri[7:], group='glance_store')
            staging_store = backend._load_store(conf, 'file')
            try:
                staging_store.configure()
            except AttributeError:
                msg = _("'node_staging_uri' is not set correctly. Could not load staging store.")
                raise exception.BadStoreUri(message=msg)
            return staging_store
        if CONF.enabled_backends:
            staging_store = glance_store.get_store_from_store_identifier('os_glance_staging_store')
        else:
            staging_store = _build_staging_store()
        try:
            image.status = 'uploading'
            image_repo.save(image, from_state='queued')
            ks_quota.enforce_image_count_uploading(req.context, req.context.owner)
            try:
                uri, size, id, store_info = staging_store.add(image_id, utils.LimitingReader(utils.CooperativeReader(data), CONF.image_size_cap), 0)
                image.size = size
            except glance_store.Duplicate:
                msg = _('The image %s has data on staging') % image_id
                raise webob.exc.HTTPConflict(explanation=msg)
            self_url = CONF.worker_self_reference_url or CONF.public_endpoint
            if self_url:
                image.extra_properties['os_glance_stage_host'] = self_url
            image_repo.save(image, from_state='uploading')
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except glance_store.StorageFull as e:
            msg = _('Image storage media is full: %s') % encodeutils.exception_to_unicode(e)
            LOG.error(msg)
            self._unstage(image_repo, image, staging_store)
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=msg, request=req)
        except exception.StorageQuotaFull as e:
            msg = _('Image exceeds the storage quota: %s') % encodeutils.exception_to_unicode(e)
            LOG.debug(msg)
            self._unstage(image_repo, image, staging_store)
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=msg, request=req)
        except exception.ImageSizeLimitExceeded as e:
            msg = _('The incoming image is too large: %s') % encodeutils.exception_to_unicode(e)
            LOG.debug(msg)
            self._unstage(image_repo, image, staging_store)
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=msg, request=req)
        except exception.LimitExceeded as e:
            LOG.debug(str(e))
            self._unstage(image_repo, image, staging_store)
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=str(e), request=req)
        except glance_store.StorageWriteDenied as e:
            msg = _('Insufficient permissions on image storage media: %s') % encodeutils.exception_to_unicode(e)
            LOG.error(msg)
            self._unstage(image_repo, image, staging_store)
            raise webob.exc.HTTPServiceUnavailable(explanation=msg, request=req)
        except exception.InvalidImageStatusTransition as e:
            msg = encodeutils.exception_to_unicode(e)
            LOG.debug(msg)
            raise webob.exc.HTTPConflict(explanation=e.msg, request=req)
        except Exception:
            with excutils.save_and_reraise_exception():
                LOG.exception(_LE('Failed to stage image data due to internal error'))
                self._restore(image_repo, image)

    def download(self, req, image_id):
        image_repo = self.gateway.get_repo(req.context)
        try:
            image = image_repo.get(image_id)
            if image.status == 'deactivated' and (not req.context.is_admin):
                msg = _('The requested image has been deactivated. Image data download is forbidden.')
                raise exception.Forbidden(message=msg)
            api_pol = api_policy.ImageAPIPolicy(req.context, image, self.policy)
            api_pol.download_image()
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to download image '%s'", image_id)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        return image