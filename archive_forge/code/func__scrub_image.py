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