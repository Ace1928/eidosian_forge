import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class UnsupportedTokenVersionException(UnexpectedError):
    debug_message_format = _('Token version is unrecognizable or unsupported.')