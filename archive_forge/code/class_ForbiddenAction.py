import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class ForbiddenAction(Forbidden):
    message_format = _('You are not authorized to perform the requested action: %(action)s.')