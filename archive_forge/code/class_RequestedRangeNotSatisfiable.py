import inspect
import sys
from magnumclient.i18n import _
class RequestedRangeNotSatisfiable(HTTPClientError):
    """HTTP 416 - Requested Range Not Satisfiable.

    The client has asked for a portion of the file, but the server cannot
    supply that portion.
    """
    http_status = 416
    message = _('Requested Range Not Satisfiable')