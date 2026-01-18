import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class InvalidLimit(Forbidden):
    message_format = _('Invalid resource limit: %(reason)s.')