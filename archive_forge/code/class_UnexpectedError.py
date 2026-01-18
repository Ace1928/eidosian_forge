import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class UnexpectedError(SecurityError):
    """Avoids exposing details of failures, unless in insecure_debug mode."""
    message_format = _('An unexpected error prevented the server from fulfilling your request.')
    debug_message_format = _('An unexpected error prevented the server from fulfilling your request: %(exception)s.')

    def _build_message(self, message, **kwargs):
        kwargs.setdefault('exception', '')
        return super(UnexpectedError, self)._build_message(message or self.debug_message_format, **kwargs)
    code = int(http.client.INTERNAL_SERVER_ERROR)
    title = http.client.responses[http.client.INTERNAL_SERVER_ERROR]