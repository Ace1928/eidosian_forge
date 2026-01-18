import inspect
import sys
from magnumclient.i18n import _
class HttpNotImplemented(HttpServerError):
    """HTTP 501 - Not Implemented.

    The server either does not recognize the request method, or it lacks
    the ability to fulfill the request.
    """
    http_status = 501
    message = _('Not Implemented')