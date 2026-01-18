import inspect
import sys
from magnumclient.i18n import _
class Unauthorized(HTTPClientError):
    """HTTP 401 - Unauthorized.

    Similar to 403 Forbidden, but specifically for use when authentication
    is required and has failed or has not yet been provided.
    """
    http_status = 401
    message = _('Unauthorized')