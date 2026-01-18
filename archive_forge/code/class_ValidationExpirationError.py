import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class ValidationExpirationError(Error):
    message_format = _("The 'expires_at' must not be before now. The server could not comply with the request since it is either malformed or otherwise incorrect. The client is assumed to be in error.")
    code = int(http.client.BAD_REQUEST)
    title = http.client.responses[http.client.BAD_REQUEST]