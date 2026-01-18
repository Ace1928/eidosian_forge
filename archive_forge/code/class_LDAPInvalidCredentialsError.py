import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class LDAPInvalidCredentialsError(UnexpectedError):
    message_format = _('Unable to authenticate against Identity backend - Invalid username or password')