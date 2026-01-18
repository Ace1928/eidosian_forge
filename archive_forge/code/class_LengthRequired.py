import inspect
import sys
from magnumclient.i18n import _
class LengthRequired(HTTPClientError):
    """HTTP 411 - Length Required.

    The request did not specify the length of its content, which is
    required by the requested resource.
    """
    http_status = 411
    message = _('Length Required')