import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class GroupNotFound(NotFound):
    message_format = _('Could not find group: %(group_id)s.')