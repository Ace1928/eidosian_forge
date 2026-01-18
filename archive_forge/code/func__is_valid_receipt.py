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
def _is_valid_receipt(self, receipt, window_seconds=0):
    """Verify the receipt is valid format and has not expired."""
    current_time = timeutils.normalize_time(timeutils.utcnow())
    try:
        expiry = timeutils.parse_isotime(receipt.expires_at)
        expiry = timeutils.normalize_time(expiry)
        expiry += datetime.timedelta(seconds=window_seconds)
    except Exception:
        LOG.exception('Unexpected error or malformed receipt determining receipt expiry: %s', receipt)
        raise exception.ReceiptNotFound(_('Failed to validate receipt'), receipt_id=receipt.id)
    if current_time < expiry:
        return None
    else:
        raise exception.ReceiptNotFound(_('Failed to validate receipt'), receipt_id=receipt.id)