import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class DirectMappingError(UnexpectedError):
    debug_message_format = _("Local section in mapping %(mapping_id)s refers to a remote match that doesn't exist (e.g. {0} in a local section).")