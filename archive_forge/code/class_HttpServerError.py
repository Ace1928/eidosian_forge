import inspect
import sys
from magnumclient.i18n import _
class HttpServerError(HttpError):
    """Server-side HTTP error.

    Exception for cases in which the server is aware that it has
    erred or is incapable of performing the request.
    """
    message = _('HTTP Server Error')