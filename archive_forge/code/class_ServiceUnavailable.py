import inspect
import sys
from magnumclient.i18n import _
class ServiceUnavailable(HttpServerError):
    """HTTP 503 - Service Unavailable.

    The server is currently unavailable.
    """
    http_status = 503
    message = _('Service Unavailable')