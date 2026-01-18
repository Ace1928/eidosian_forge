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
def _get_images_page(self, marker):
    filters = {'deleted': True, 'status': 'pending_delete'}
    return db_api.get_api().image_get_all(self.admin_context, filters=filters, marker=marker, limit=REASONABLE_DB_PAGE_SIZE)