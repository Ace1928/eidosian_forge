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