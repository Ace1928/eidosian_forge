import inspect
import sys
from magnumclient.i18n import _
class HttpVersionNotSupported(HttpServerError):
    """HTTP 505 - HttpVersion Not Supported.

    The server does not support the HTTP protocol version used in the request.
    """
    http_status = 505
    message = _('HTTP Version Not Supported')