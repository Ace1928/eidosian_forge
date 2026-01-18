import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class LDAPServerConnectionError(UnexpectedError):
    debug_message_format = _('Unable to establish a connection to LDAP Server (%(url)s).')