import datetime
import hashlib
import http.client as http
import os
import re
import urllib.parse as urlparse
import uuid
from castellan.common import exception as castellan_exception
from castellan import key_manager
import glance_store
from glance_store import location
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from oslo_utils import encodeutils
from oslo_utils import timeutils as oslo_timeutils
import requests
import webob.exc
from glance.api import common
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
from glance import context as glance_context
import glance.db
import glance.gateway
from glance.i18n import _, _LE, _LI, _LW
import glance.notifier
from glance.quota import keystone as ks_quota
import glance.schema
class ImagesController(object):

    def __init__(self, db_api=None, policy_enforcer=None, notifier=None, store_api=None):
        self.db_api = db_api or glance.db.get_api()
        self.policy = policy_enforcer or policy.Enforcer()
        self.notifier = notifier or glance.notifier.Notifier()
        self.store_api = store_api or glance_store
        self.gateway = glance.gateway.Gateway(self.db_api, self.store_api, self.notifier, self.policy)
        self._key_manager = key_manager.API(CONF)

    @utils.mutating
    def create(self, req, image, extra_properties, tags):
        image_factory = self.gateway.get_image_factory(req.context)
        image_repo = self.gateway.get_repo(req.context)
        try:
            if 'owner' not in image:
                image['owner'] = req.context.project_id
            api_policy.ImageAPIPolicy(req.context, image, self.policy).add_image()
            ks_quota.enforce_image_count_total(req.context, req.context.owner)
            image = image_factory.new_image(extra_properties=extra_properties, tags=tags, **image)
            image_repo.add(image)
        except (exception.DuplicateLocation, exception.Invalid) as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)
        except (exception.ReservedProperty, exception.ReadonlyProperty) as e:
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.Forbidden as e:
            LOG.debug('User not permitted to create image')
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.LimitExceeded as e:
            LOG.warning(encodeutils.exception_to_unicode(e))
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=e.msg, request=req, content_type='text/plain')
        except exception.Duplicate as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        except exception.NotAuthenticated as e:
            raise webob.exc.HTTPUnauthorized(explanation=e.msg)
        except TypeError as e:
            LOG.debug(encodeutils.exception_to_unicode(e))
            raise webob.exc.HTTPBadRequest(explanation=e)
        return image

    def _bust_import_lock(self, admin_image_repo, admin_task_repo, image, task, task_id):
        if task:
            try:
                task.fail('Expired lock preempted')
                admin_task_repo.save(task)
            except exception.InvalidTaskStatusTransition:
                pass
        try:
            admin_image_repo.delete_property_atomic(image, 'os_glance_import_task', task_id)
        except exception.NotFound:
            LOG.warning('Image %(image)s has stale import task %(task)s but we lost the race to remove it.', {'image': image.image_id, 'task': task_id})
            raise exception.Conflict('Image has active task')
        LOG.warning('Image %(image)s has stale import task %(task)s in status %(status)s from %(owner)s; removed lock because it had expired.', {'image': image.image_id, 'task': task_id, 'status': task and task.status or 'missing', 'owner': task and task.owner or 'unknown owner'})

    def _enforce_import_lock(self, req, image):
        admin_context = req.context.elevated()
        admin_image_repo = self.gateway.get_repo(admin_context)
        admin_task_repo = self.gateway.get_task_repo(admin_context)
        other_task = image.extra_properties['os_glance_import_task']
        expiry = datetime.timedelta(minutes=60)
        bustable_states = ('pending', 'processing', 'success', 'failure')
        try:
            task = admin_task_repo.get(other_task)
        except exception.NotFound:
            LOG.warning('Image %(image)s has non-existent import task %(task)s; considering it stale', {'image': image.image_id, 'task': other_task})
            task = None
            age = 0
        else:
            age = oslo_timeutils.utcnow() - task.updated_at
            if task.status == 'pending':
                expiry *= 2
        if not task or (task.status in bustable_states and age >= expiry):
            self._bust_import_lock(admin_image_repo, admin_task_repo, image, task, other_task)
            return task
        if task.status in bustable_states:
            LOG.warning('Image %(image)s has active import task %(task)s in status %(status)s; lock remains valid for %(expire)i more seconds', {'image': image.image_id, 'task': task.task_id, 'status': task.status, 'expire': (expiry - age).total_seconds()})
        else:
            LOG.debug('Image %(image)s has import task %(task)s in status %(status)s and does not qualify for expiry.', {'image': image.image_id, 'task': task.task_id, 'status': task.status})
        raise exception.Conflict('Image has active task')

    def _cleanup_stale_task_progress(self, image_repo, image, task):
        """Cleanup stale in-progress information from a previous task.

        If we stole the lock from another task, we should try to clean up
        the in-progress status information from that task while we have
        the lock.
        """
        stores = task.task_input.get('backend', [])
        keys = ['os_glance_importing_to_stores', 'os_glance_failed_import']
        changed = set()
        for store in stores:
            for key in keys:
                values = image.extra_properties.get(key, '').split(',')
                if store in values:
                    values.remove(store)
                    changed.add(key)
                image.extra_properties[key] = ','.join(values)
        if changed:
            image_repo.save(image)
            LOG.debug('Image %(image)s had stale import progress info %(keys)s from task %(task)s which was cleaned up', {'image': image.image_id, 'task': task.task_id, 'keys': ','.join(changed)})

    def _proxy_request_to_stage_host(self, image, req, body=None):
        """Proxy a request to a staging host.

        When an image was staged on another worker, that worker may record its
        worker_self_reference_url on the image, indicating that other workers
        should proxy requests to it while the image is staged. This method
        replays our current request against the remote host, returns the
        result, and performs any response error translation required.

        The remote request-id is used to replace the one on req.context so that
        a client sees the proper id used for the actual action.

        :param image: The Image from the repo
        :param req: The webob.Request from the current request
        :param body: The request body or None
        :returns: The result from the remote host
        :raises: webob.exc.HTTPClientError matching the remote's error, or
                 webob.exc.HTTPServerError if we were unable to contact the
                 remote host.
        """
        stage_host = image.extra_properties['os_glance_stage_host']
        LOG.info(_LI('Proxying %s request to host %s which has image staged'), req.method, stage_host)
        client = glance_context.get_ksa_client(req.context)
        url = '%s%s' % (stage_host, req.path)
        req_id_hdr = 'x-openstack-request-id'
        request_method = getattr(client, req.method.lower())
        try:
            r = request_method(url, json=body, timeout=60)
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as e:
            LOG.error(_LE('Failed to proxy to %r: %s'), url, e)
            raise webob.exc.HTTPGatewayTimeout('Stage host is unavailable')
        except requests.exceptions.RequestException as e:
            LOG.error(_LE('Failed to proxy to %r: %s'), url, e)
            raise webob.exc.HTTPBadGateway('Stage host is unavailable')
        req_id_hdr = 'x-openstack-request-id'
        if req_id_hdr in r.headers:
            LOG.debug('Replying with remote request id %s', r.headers[req_id_hdr])
            req.context.request_id = r.headers[req_id_hdr]
        if r.status_code // 100 != 2:
            raise proxy_response_error(r.status_code, r.reason)
        return image.image_id

    @property
    def self_url(self):
        """Return the URL we expect to point to us.

        If this is set to a per-worker URL in worker_self_reference_url,
        that takes precedence. Otherwise we fall back to public_endpoint.
        """
        return CONF.worker_self_reference_url or CONF.public_endpoint

    def is_proxyable(self, image):
        """Decide if an action is proxyable to a stage host.

        If the image has a staging host recorded with a URL that does not match
        ours, then we can proxy our request to that host.

        :param image: The Image from the repo
        :returns: bool indicating proxyable status
        """
        return 'os_glance_stage_host' in image.extra_properties and image.extra_properties['os_glance_stage_host'] != self.self_url

    @utils.mutating
    def import_image(self, req, image_id, body):
        ctxt = req.context
        image_repo = self.gateway.get_repo(ctxt)
        task_factory = self.gateway.get_task_factory(ctxt)
        task_repo = self.gateway.get_task_repo(ctxt)
        import_method = body.get('method').get('name')
        uri = body.get('method').get('uri')
        all_stores_must_succeed = body.get('all_stores_must_succeed', True)
        stole_lock_from_task = None
        try:
            ks_quota.enforce_image_size_total(req.context, req.context.owner)
        except exception.LimitExceeded as e:
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=str(e), request=req)
        try:
            image = image_repo.get(image_id)
            if image.status == 'active' and import_method != 'copy-image':
                msg = _('Image with status active cannot be target for import')
                raise exception.Conflict(msg)
            if image.status != 'active' and import_method == 'copy-image':
                msg = _('Only images with status active can be targeted for copying')
                raise exception.Conflict(msg)
            if image.status != 'queued' and import_method in ['web-download', 'glance-download']:
                msg = _("Image needs to be in 'queued' state to use '%s' method") % import_method
                raise exception.Conflict(msg)
            if image.status != 'uploading' and import_method == 'glance-direct':
                msg = _("Image needs to be staged before 'glance-direct' method can be used")
                raise exception.Conflict(msg)
            if not getattr(image, 'container_format', None):
                msg = _("'container_format' needs to be set before import")
                raise exception.Conflict(msg)
            if not getattr(image, 'disk_format', None):
                msg = _("'disk_format' needs to be set before import")
                raise exception.Conflict(msg)
            if import_method == 'glance-download':
                if 'glance_region' not in body.get('method'):
                    msg = _("'glance_region' needs to be set for glance-download import method")
                    raise webob.exc.HTTPBadRequest(explanation=msg)
                if 'glance_image_id' not in body.get('method'):
                    msg = _("'glance_image_id' needs to be set for glance-download import method")
                    raise webob.exc.HTTPBadRequest(explanation=msg)
                try:
                    uuid.UUID(body['method']['glance_image_id'])
                except ValueError:
                    msg = _('Remote image id does not look like a UUID: %s') % body['method']['glance_image_id']
                    raise webob.exc.HTTPBadRequest(explanation=msg)
                if 'glance_service_interface' not in body.get('method'):
                    body.get('method')['glance_service_interface'] = 'public'
            api_pol = api_policy.ImageAPIPolicy(req.context, image, enforcer=self.policy)
            if import_method == 'copy-image':
                api_pol.copy_image()
            else:
                api_pol.modify_image()
            if 'os_glance_import_task' in image.extra_properties:
                stole_lock_from_task = self._enforce_import_lock(req, image)
            stores = [None]
            if CONF.enabled_backends:
                try:
                    stores = utils.get_stores_from_request(req, body)
                except glance_store.UnknownScheme as exc:
                    LOG.warning(exc.msg)
                    raise exception.Conflict(exc.msg)
            all_stores = body.get('all_stores', False)
            if import_method == 'copy-image' and all_stores:
                for loc in image.locations:
                    existing_store = loc['metadata']['store']
                    if existing_store in stores:
                        LOG.debug("Removing store '%s' from all stores as image is already available in that store.", existing_store)
                        stores.remove(existing_store)
                if len(stores) == 0:
                    LOG.info(_LI('Exiting copying workflow as image is available in all configured stores.'))
                    return image_id
            if import_method == 'copy-image' and (not all_stores):
                for loc in image.locations:
                    existing_store = loc['metadata']['store']
                    if existing_store in stores:
                        msg = _("Image is already present at store '%s'") % existing_store
                        raise webob.exc.HTTPBadRequest(explanation=msg)
        except exception.Conflict as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Forbidden as e:
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        if not all_stores_must_succeed and (not CONF.enabled_backends):
            msg = _('All_stores_must_succeed can only be set with enabled_backends %s') % uri
            raise webob.exc.HTTPBadRequest(explanation=msg)
        if self.is_proxyable(image) and import_method == 'glance-direct':
            return self._proxy_request_to_stage_host(image, req, body)
        task_input = {'image_id': image_id, 'import_req': body, 'backend': stores}
        if import_method == 'copy-image':
            admin_context = ctxt.elevated()
        else:
            admin_context = None
        executor_factory = self.gateway.get_task_executor_factory(ctxt, admin_context=admin_context)
        if import_method == 'web-download' and (not utils.validate_import_uri(uri)):
            LOG.debug('URI for web-download does not pass filtering: %s', uri)
            msg = _('URI for web-download does not pass filtering: %s') % uri
            raise webob.exc.HTTPBadRequest(explanation=msg)
        try:
            import_task = task_factory.new_task(task_type='api_image_import', owner=ctxt.owner, task_input=task_input, image_id=image_id, user_id=ctxt.user_id, request_id=ctxt.request_id)
            try:
                image_repo.set_property_atomic(image, 'os_glance_import_task', import_task.task_id)
            except exception.Duplicate:
                msg = _("New operation on image '%s' is not permitted as prior operation is still in progress") % image_id
                raise exception.Conflict(msg)
            if stole_lock_from_task:
                self._cleanup_stale_task_progress(image_repo, image, stole_lock_from_task)
            task_repo.add(import_task)
            task_executor = executor_factory.new_task_executor(ctxt)
            pool = common.get_thread_pool('tasks_pool')
            pool.spawn(import_task.run, task_executor)
        except exception.Forbidden as e:
            LOG.debug('User not permitted to create image import task.')
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.Conflict as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        except exception.InvalidImageStatusTransition as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        except exception.LimitExceeded as e:
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=str(e), request=req)
        except ValueError as e:
            LOG.debug('Cannot import data for image %(id)s: %(e)s', {'id': image_id, 'e': encodeutils.exception_to_unicode(e)})
            raise webob.exc.HTTPBadRequest(explanation=encodeutils.exception_to_unicode(e))
        return image_id

    def index(self, req, marker=None, limit=None, sort_key=None, sort_dir=None, filters=None, member_status='accepted'):
        sort_key = ['created_at'] if not sort_key else sort_key
        sort_dir = ['desc'] if not sort_dir else sort_dir
        result = {}
        if filters is None:
            filters = {}
        filters['deleted'] = False
        os_hidden = filters.get('os_hidden', 'false').lower()
        if os_hidden not in ['true', 'false']:
            message = _("Invalid value '%s' for 'os_hidden' filter. Valid values are 'true' or 'false'.") % os_hidden
            raise webob.exc.HTTPBadRequest(explanation=message)
        filters['os_hidden'] = os_hidden == 'true'
        protected = filters.get('protected')
        if protected is not None:
            if protected not in ['true', 'false']:
                message = _("Invalid value '%s' for 'protected' filter. Valid values are 'true' or 'false'.") % protected
                raise webob.exc.HTTPBadRequest(explanation=message)
            filters['protected'] = protected == 'true'
        if limit is None:
            limit = CONF.limit_param_default
        limit = min(CONF.api_limit_max, limit)
        image_repo = self.gateway.get_repo(req.context)
        try:
            target = {'project_id': req.context.project_id}
            self.policy.enforce(req.context, 'get_images', target)
            images = image_repo.list(marker=marker, limit=limit, sort_key=sort_key, sort_dir=sort_dir, filters=filters, member_status=member_status)
            db_image_count = len(images)
            images = [image for image in images if api_policy.ImageAPIPolicy(req.context, image, self.policy).check('get_image')]
            if len(images) != 0 and db_image_count == limit:
                result['next_marker'] = images[-1].image_id
        except (exception.NotFound, exception.InvalidSortKey, exception.InvalidFilterRangeValue, exception.InvalidParameterValue, exception.InvalidFilterOperatorValue) as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)
        except exception.Forbidden as e:
            LOG.debug('User not permitted to retrieve images index')
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotAuthenticated as e:
            raise webob.exc.HTTPUnauthorized(explanation=e.msg)
        result['images'] = images
        return result

    def show(self, req, image_id):
        image_repo = self.gateway.get_repo(req.context)
        try:
            image = image_repo.get(image_id)
            api_policy.ImageAPIPolicy(req.context, image, self.policy).get_image()
            return image
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.NotAuthenticated as e:
            raise webob.exc.HTTPUnauthorized(explanation=e.msg)

    def get_task_info(self, req, image_id):
        image_repo = self.gateway.get_repo(req.context)
        try:
            image = image_repo.get(image_id)
            api_policy.ImageAPIPolicy(req.context, image, self.policy).get_image()
        except (exception.NotFound, exception.Forbidden):
            raise webob.exc.HTTPNotFound()
        tasks = self.db_api.tasks_get_by_image(req.context, image.image_id)
        return {'tasks': tasks}

    @utils.mutating
    def update(self, req, image_id, changes):
        image_repo = self.gateway.get_repo(req.context)
        try:
            image = image_repo.get(image_id)
            api_pol = api_policy.ImageAPIPolicy(req.context, image, self.policy)
            for change in changes:
                change_method_name = '_do_%s' % change['op']
                change_method = getattr(self, change_method_name)
                change_method(req, image, api_pol, change)
            if changes:
                image_repo.save(image)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except (exception.Invalid, exception.BadStoreUri) as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to update image '%s'", image_id)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.StorageQuotaFull as e:
            msg = _('Denying attempt to upload image because it exceeds the quota: %s') % encodeutils.exception_to_unicode(e)
            LOG.warning(msg)
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=msg, request=req, content_type='text/plain')
        except exception.LimitExceeded as e:
            LOG.exception(encodeutils.exception_to_unicode(e))
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=e.msg, request=req, content_type='text/plain')
        except exception.NotAuthenticated as e:
            raise webob.exc.HTTPUnauthorized(explanation=e.msg)
        return image

    def _do_replace(self, req, image, api_pol, change):
        path = change['path']
        path_root = path[0]
        value = change['value']
        if path_root == 'locations' and (not value):
            msg = _('Cannot set locations to empty list.')
            raise webob.exc.HTTPForbidden(msg)
        elif path_root == 'locations' and value:
            api_pol.update_locations()
            self._do_replace_locations(image, value)
        elif path_root == 'owner' and req.context.is_admin == False:
            msg = _("Owner can't be updated by non admin.")
            raise webob.exc.HTTPForbidden(msg)
        else:
            api_pol.update_property(path_root, value)
            if hasattr(image, path_root):
                setattr(image, path_root, value)
            elif path_root in image.extra_properties:
                image.extra_properties[path_root] = value
            else:
                msg = _('Property %s does not exist.')
                raise webob.exc.HTTPConflict(msg % path_root)

    def _do_add(self, req, image, api_pol, change):
        path = change['path']
        path_root = path[0]
        value = change['value']
        json_schema_version = change.get('json_schema_version', 10)
        if path_root == 'locations':
            api_pol.update_locations()
            self._do_add_locations(image, path[1], value)
        else:
            api_pol.update_property(path_root, value)
            if (hasattr(image, path_root) or path_root in image.extra_properties) and json_schema_version == 4:
                msg = _('Property %s already present.')
                raise webob.exc.HTTPConflict(msg % path_root)
            if hasattr(image, path_root):
                setattr(image, path_root, value)
            else:
                image.extra_properties[path_root] = value

    def _do_remove(self, req, image, api_pol, change):
        path = change['path']
        path_root = path[0]
        if path_root == 'locations':
            api_pol.delete_locations()
            try:
                self._do_remove_locations(image, path[1])
            except exception.Forbidden as e:
                raise webob.exc.HTTPForbidden(e.msg)
        else:
            api_pol.update_property(path_root)
            if hasattr(image, path_root):
                msg = _('Property %s may not be removed.')
                raise webob.exc.HTTPForbidden(msg % path_root)
            elif path_root in image.extra_properties:
                del image.extra_properties[path_root]
            else:
                msg = _('Property %s does not exist.')
                raise webob.exc.HTTPConflict(msg % path_root)

    def _delete_encryption_key(self, context, image):
        props = image.extra_properties
        cinder_encryption_key_id = props.get('cinder_encryption_key_id')
        if cinder_encryption_key_id is None:
            return
        deletion_policy = props.get('cinder_encryption_key_deletion_policy', '')
        if deletion_policy != 'on_image_deletion':
            return
        try:
            self._key_manager.delete(context, cinder_encryption_key_id)
        except castellan_exception.Forbidden:
            msg = 'Not allowed to delete encryption key %s' % cinder_encryption_key_id
            LOG.warning(msg)
        except (castellan_exception.ManagedObjectNotFoundError, KeyError):
            msg = 'Could not find encryption key %s' % cinder_encryption_key_id
            LOG.warning(msg)
        except castellan_exception.KeyManagerError:
            msg = 'Failed to delete cinder encryption key %s' % cinder_encryption_key_id
            LOG.warning(msg)

    @utils.mutating
    def delete_from_store(self, req, store_id, image_id):
        if not CONF.enabled_backends:
            raise webob.exc.HTTPNotFound()
        if store_id not in CONF.enabled_backends:
            msg = _('The selected store %s is not available on this node.') % store_id
            raise webob.exc.HTTPConflict(explanation=msg)
        image_repo = self.gateway.get_repo(req.context)
        try:
            image = image_repo.get(image_id)
        except exception.NotAuthenticated as e:
            raise webob.exc.HTTPUnauthorized(explanation=e.msg)
        except exception.NotFound:
            msg = _('Failed to find image %(image_id)s') % {'image_id': image_id}
            raise webob.exc.HTTPNotFound(explanation=msg)
        api_pol = api_policy.ImageAPIPolicy(req.context, image, self.policy)
        api_pol.get_image_location()
        try:
            api_pol.delete_locations()
        except exception.Forbidden as e:
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        if image.status != 'active':
            msg = _("It's not allowed to remove image data from store if image status is not 'active'")
            raise webob.exc.HTTPConflict(explanation=msg)
        if len(image.locations) == 1:
            LOG.debug('User forbidden to remove last location of image %s', image_id)
            msg = _('Cannot delete image data from the only store containing it. Consider deleting the image instead.')
            raise webob.exc.HTTPForbidden(explanation=msg)
        try:
            for pos, loc in enumerate(image.locations):
                if loc['metadata'].get('store') == store_id:
                    image.locations.pop(pos)
                    break
            else:
                msg = _('Image %(iid)s is not stored in store %(sid)s.') % {'iid': image_id, 'sid': store_id}
                raise exception.Invalid(msg)
        except exception.Forbidden as e:
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.Invalid as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except glance_store.exceptions.HasSnapshot as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        except glance_store.exceptions.InUseByStore as e:
            msg = 'The data for Image %(id)s could not be deleted because it is in use: %(exc)s' % {'id': image_id, 'exc': e.msg}
            LOG.warning(msg)
            raise webob.exc.HTTPConflict(explanation=msg)
        except Exception as e:
            raise webob.exc.HTTPInternalServerError(explanation=encodeutils.exception_to_unicode(e))
        image_repo.save(image)

    def _delete_image_on_remote(self, image, req):
        """Proxy an image delete to a staging host.

        When an image is staged and then deleted, the staging host still
        has local residue that needs to be cleaned up. If the request to
        delete arrived here, but we are not the stage host, we need to
        proxy it to the appropriate host.

        If the delete succeeds, we return None (per DELETE semantics),
        indicating to the caller that it was handled.

        If the delete fails on the remote end, we allow the
        HTTPClientError to bubble to our caller, which will return the
        error to the client.

        If we fail to contact the remote server, we catch the
        HTTPServerError raised by our proxy method, verify that the
        image still exists, and return it. That indicates to the
        caller that it should proceed with the regular delete logic,
        which will satisfy the client's request, but leave the residue
        on the stage host (which is unavoidable).

        :param image: The Image from the repo
        :param req: The webob.Request for this call
        :returns: None if successful, or a refreshed image if the proxy failed.
        :raises: webob.exc.HTTPClientError if so raised by the remote server.
        """
        try:
            self._proxy_request_to_stage_host(image, req)
        except webob.exc.HTTPServerError:
            return self.gateway.get_repo(req.context).get(image.image_id)

    @utils.mutating
    def delete(self, req, image_id):
        image_repo = self.gateway.get_repo(req.context)
        try:
            image = image_repo.get(image_id)
            api_pol = api_policy.ImageAPIPolicy(req.context, image, self.policy)
            api_pol.delete_image()
            if self.is_proxyable(image):
                image = self._delete_image_on_remote(image, req)
                if image is None:
                    return
            if CONF.enabled_backends:
                separator, staging_dir = store_utils.get_dir_separator()
                file_path = '%s%s%s' % (staging_dir, separator, image_id)
                try:
                    fn_call = glance_store.get_store_from_store_identifier
                    staging_store = fn_call('os_glance_staging_store')
                    loc = location.get_location_from_uri_and_backend(file_path, 'os_glance_staging_store')
                    staging_store.delete(loc)
                except (glance_store.exceptions.NotFound, glance_store.exceptions.UnknownScheme):
                    pass
            else:
                file_path = str(CONF.node_staging_uri + '/' + image_id)[7:]
                if os.path.exists(file_path):
                    try:
                        LOG.debug('After upload to the backend, deleting staged image data from %(fn)s', {'fn': file_path})
                        os.unlink(file_path)
                    except OSError as e:
                        LOG.error('After upload to backend, deletion of staged image data from %(fn)s has failed because [Errno %(en)d]', {'fn': file_path, 'en': e.errno})
                else:
                    LOG.warning(_('After upload to backend, deletion of staged image data has failed because it cannot be found at %(fn)s'), {'fn': file_path})
            image.delete()
            self._delete_encryption_key(req.context, image)
            image_repo.remove(image)
        except (glance_store.Forbidden, exception.Forbidden) as e:
            LOG.debug("User not permitted to delete image '%s'", image_id)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except (glance_store.NotFound, exception.NotFound):
            msg = _('Failed to find image %(image_id)s to delete') % {'image_id': image_id}
            LOG.warning(msg)
            raise webob.exc.HTTPNotFound(explanation=msg)
        except glance_store.exceptions.InUseByStore as e:
            msg = _('Image %(id)s could not be deleted because it is in use: %(exc)s') % {'id': image_id, 'exc': e.msg}
            LOG.warning(msg)
            raise webob.exc.HTTPConflict(explanation=msg)
        except glance_store.exceptions.HasSnapshot as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        except exception.InvalidImageStatusTransition as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)
        except exception.NotAuthenticated as e:
            raise webob.exc.HTTPUnauthorized(explanation=e.msg)

    def _validate_validation_data(self, image, locations):
        val_data = {}
        for loc in locations:
            if 'validation_data' not in loc:
                continue
            for k, v in loc['validation_data'].items():
                if val_data.get(k, v) != v:
                    msg = _('Conflicting values for %s') % k
                    raise webob.exc.HTTPConflict(explanation=msg)
                val_data[k] = v
        new_val_data = {}
        for k, v in val_data.items():
            current = getattr(image, k)
            if v == current:
                continue
            if current:
                msg = _('%s is already set with a different value') % k
                raise webob.exc.HTTPConflict(explanation=msg)
            new_val_data[k] = v
        if not new_val_data:
            return {}
        if image.status != 'queued':
            msg = _("New value(s) for %s may only be provided when image status is 'queued'") % ', '.join(new_val_data.keys())
            raise webob.exc.HTTPConflict(explanation=msg)
        if 'checksum' in new_val_data:
            try:
                checksum_bytes = bytearray.fromhex(new_val_data['checksum'])
            except ValueError:
                msg = _('checksum (%s) is not a valid hexadecimal value') % new_val_data['checksum']
                raise webob.exc.HTTPConflict(explanation=msg)
            if len(checksum_bytes) != 16:
                msg = _('checksum (%s) is not the correct size for md5 (should be 16 bytes)') % new_val_data['checksum']
                raise webob.exc.HTTPConflict(explanation=msg)
        hash_algo = new_val_data.get('os_hash_algo')
        if hash_algo != CONF['hashing_algorithm']:
            msg = _('os_hash_algo must be %(want)s, not %(got)s') % {'want': CONF['hashing_algorithm'], 'got': hash_algo}
            raise webob.exc.HTTPConflict(explanation=msg)
        try:
            hash_bytes = bytearray.fromhex(new_val_data['os_hash_value'])
        except ValueError:
            msg = _('os_hash_value (%s) is not a valid hexadecimal value') % new_val_data['os_hash_value']
            raise webob.exc.HTTPConflict(explanation=msg)
        want_size = hashlib.new(hash_algo).digest_size
        if len(hash_bytes) != want_size:
            msg = _('os_hash_value (%(value)s) is not the correct size for %(algo)s (should be %(want)d bytes)') % {'value': new_val_data['os_hash_value'], 'algo': hash_algo, 'want': want_size}
            raise webob.exc.HTTPConflict(explanation=msg)
        return new_val_data

    def _get_locations_op_pos(self, path_pos, max_pos, allow_max):
        if path_pos is None or max_pos is None:
            return None
        pos = max_pos if allow_max else max_pos - 1
        if path_pos.isdigit():
            pos = int(path_pos)
        elif path_pos != '-':
            return None
        if not (allow_max or 0 <= pos < max_pos):
            return None
        return pos

    def _do_replace_locations(self, image, value):
        if CONF.show_multiple_locations == False:
            msg = _("It's not allowed to update locations if locations are invisible.")
            raise webob.exc.HTTPForbidden(explanation=msg)
        if image.status not in ('active', 'queued'):
            msg = _("It's not allowed to replace locations if image status is %s.") % image.status
            raise webob.exc.HTTPConflict(explanation=msg)
        val_data = self._validate_validation_data(image, value)
        updated_location = value
        if CONF.enabled_backends:
            updated_location = store_utils.get_updated_store_location(value)
        try:
            image.locations = updated_location
            if image.status == 'queued':
                for k, v in val_data.items():
                    setattr(image, k, v)
                image.status = 'active'
        except (exception.BadStoreUri, exception.DuplicateLocation) as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)
        except ValueError as ve:
            raise webob.exc.HTTPBadRequest(explanation=encodeutils.exception_to_unicode(ve))

    def _do_add_locations(self, image, path_pos, value):
        if CONF.show_multiple_locations == False:
            msg = _("It's not allowed to add locations if locations are invisible.")
            raise webob.exc.HTTPForbidden(explanation=msg)
        if image.status not in ('active', 'queued'):
            msg = _("It's not allowed to add locations if image status is %s.") % image.status
            raise webob.exc.HTTPConflict(explanation=msg)
        val_data = self._validate_validation_data(image, [value])
        updated_location = value
        if CONF.enabled_backends:
            updated_location = store_utils.get_updated_store_location([value])[0]
        pos = self._get_locations_op_pos(path_pos, len(image.locations), True)
        if pos is None:
            msg = _('Invalid position for adding a location.')
            raise webob.exc.HTTPBadRequest(explanation=msg)
        try:
            image.locations.insert(pos, updated_location)
            if image.status == 'queued':
                for k, v in val_data.items():
                    setattr(image, k, v)
                image.status = 'active'
        except (exception.BadStoreUri, exception.DuplicateLocation) as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)
        except ValueError as e:
            raise webob.exc.HTTPBadRequest(explanation=encodeutils.exception_to_unicode(e))

    def _do_remove_locations(self, image, path_pos):
        if CONF.show_multiple_locations == False:
            msg = _("It's not allowed to remove locations if locations are invisible.")
            raise webob.exc.HTTPForbidden(explanation=msg)
        if image.status not in 'active':
            msg = _("It's not allowed to remove locations if image status is %s.") % image.status
            raise webob.exc.HTTPConflict(explanation=msg)
        if len(image.locations) == 1:
            LOG.debug('User forbidden to remove last location of image %s', image.image_id)
            msg = _('Cannot remove last location in the image.')
            raise exception.Forbidden(msg)
        pos = self._get_locations_op_pos(path_pos, len(image.locations), False)
        if pos is None:
            msg = _('Invalid position for removing a location.')
            raise webob.exc.HTTPBadRequest(explanation=msg)
        try:
            image.locations.pop(pos)
        except Exception as e:
            raise webob.exc.HTTPInternalServerError(explanation=encodeutils.exception_to_unicode(e))