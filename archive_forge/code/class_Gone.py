import inspect
import sys
from magnumclient.i18n import _
class Gone(HTTPClientError):
    """HTTP 410 - Gone.

    Indicates that the resource requested is no longer available and will
    not be available again.
    """
    http_status = 410
    message = _('Gone')