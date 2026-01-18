import calendar
import time
import eventlet
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_log import versionutils
from oslo_utils import encodeutils
from glance.common import crypt
from glance.common import exception
from glance.common import timeutils
from glance import context
import glance.db as db_api
from glance.i18n import _, _LC, _LE, _LI, _LW
class Scrubber(object):

    def __init__(self, store_api):
        versionutils.report_deprecated_feature(LOG, DEPRECATED_SCRUBBER_MSG)
        LOG.info(_LI('Initializing scrubber'))
        self.store_api = store_api
        self.admin_context = context.get_admin_context(show_deleted=True)
        self.db_queue = get_scrub_queue()
        self.pool = eventlet.greenpool.GreenPool(CONF.scrub_pool_size)

    def _get_delete_jobs(self):
        try:
            records = self.db_queue.get_all_locations()
        except Exception as err:
            msg = _LC('Can not get scrub jobs from queue: %s') % encodeutils.exception_to_unicode(err)
            LOG.critical(msg)
            raise exception.FailedToGetScrubberJobs()
        delete_jobs = {}
        if CONF.enabled_backends:
            for image_id, loc_id, loc_uri, backend in records:
                if image_id not in delete_jobs:
                    delete_jobs[image_id] = []
                delete_jobs[image_id].append((image_id, loc_id, loc_uri, backend))
        else:
            for image_id, loc_id, loc_uri in records:
                if image_id not in delete_jobs:
                    delete_jobs[image_id] = []
                delete_jobs[image_id].append((image_id, loc_id, loc_uri))
        return delete_jobs

    def run(self, event=None):
        delete_jobs = self._get_delete_jobs()
        if delete_jobs:
            list(self.pool.starmap(self._scrub_image, delete_jobs.items()))

    def _scrub_image(self, image_id, delete_jobs):
        if len(delete_jobs) == 0:
            return
        LOG.info(_LI('Scrubbing image %(id)s from %(count)d locations.'), {'id': image_id, 'count': len(delete_jobs)})
        success = True
        if CONF.enabled_backends:
            for img_id, loc_id, uri, backend in delete_jobs:
                try:
                    self._delete_image_location_from_backend(img_id, loc_id, uri, backend=backend)
                except Exception:
                    success = False
        else:
            for img_id, loc_id, uri in delete_jobs:
                try:
                    self._delete_image_location_from_backend(img_id, loc_id, uri)
                except Exception:
                    success = False
        if success:
            image = db_api.get_api().image_get(self.admin_context, image_id)
            if image['status'] == 'pending_delete':
                db_api.get_api().image_update(self.admin_context, image_id, {'status': 'deleted'})
            LOG.info(_LI('Image %s has been scrubbed successfully'), image_id)
        else:
            LOG.warning(_LW("One or more image locations couldn't be scrubbed from backend. Leaving image '%s' in 'pending_delete' status"), image_id)

    def _delete_image_location_from_backend(self, image_id, loc_id, uri, backend=None):
        try:
            LOG.debug('Scrubbing image %s from a location.', image_id)
            try:
                if CONF.enabled_backends:
                    self.store_api.delete(uri, backend, self.admin_context)
                else:
                    self.store_api.delete_from_backend(uri, self.admin_context)
            except store_exceptions.NotFound:
                LOG.info(_LI("Image location for image '%s' not found in backend; Marking image location deleted in db."), image_id)
            if loc_id != '-':
                db_api.get_api().image_location_delete(self.admin_context, image_id, int(loc_id), 'deleted')
            LOG.info(_LI('Image %s is scrubbed from a location.'), image_id)
        except Exception as e:
            LOG.error(_LE('Unable to scrub image %(id)s from a location. Reason: %(exc)s '), {'id': image_id, 'exc': encodeutils.exception_to_unicode(e)})
            raise

    def revert_image_status(self, image_id):
        db_api.get_api().image_restore(self.admin_context, image_id)