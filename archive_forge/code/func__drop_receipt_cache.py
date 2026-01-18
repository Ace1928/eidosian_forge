import datetime
from oslo_log import log
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import manager
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.models import receipt_model
from keystone import notifications
def _drop_receipt_cache(self, service, resource_type, operation, payload):
    """Invalidate the entire receipt cache.

        This is a handy private utility method that should be used when
        consuming notifications that signal invalidating the receipt cache.

        """
    if CONF.receipt.cache_on_issue:
        RECEIPTS_REGION.invalidate()