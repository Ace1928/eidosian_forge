import inspect
import sys
from magnumclient.i18n import _
class ExpectationFailed(HTTPClientError):
    """HTTP 417 - Expectation Failed.

    The server cannot meet the requirements of the Expect request-header field.
    """
    http_status = 417
    message = _('Expectation Failed')