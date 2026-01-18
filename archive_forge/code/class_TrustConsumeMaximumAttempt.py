import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class TrustConsumeMaximumAttempt(UnexpectedError):
    debug_message_format = _('Unable to consume trust %(trust_id)s. Unable to acquire lock.')