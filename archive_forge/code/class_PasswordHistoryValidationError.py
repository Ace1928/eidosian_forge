import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class PasswordHistoryValidationError(PasswordValidationError):
    message_format = _('The new password cannot be identical to a previous password. The total number which includes the new password must be unique is %(unique_count)s.')