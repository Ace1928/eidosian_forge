import inspect
import sys
from magnumclient.i18n import _
class BadGateway(HttpServerError):
    """HTTP 502 - Bad Gateway.

    The server was acting as a gateway or proxy and received an invalid
    response from the upstream server.
    """
    http_status = 502
    message = _('Bad Gateway')