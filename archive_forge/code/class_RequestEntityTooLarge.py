import inspect
import sys
from magnumclient.i18n import _
class RequestEntityTooLarge(HTTPClientError):
    """HTTP 413 - Request Entity Too Large.

    The request is larger than the server is willing or able to process.
    """
    http_status = 413
    message = _('Request Entity Too Large')

    def __init__(self, *args, **kwargs):
        try:
            self.retry_after = int(kwargs.pop('retry_after'))
        except (KeyError, ValueError):
            self.retry_after = 0
        super(RequestEntityTooLarge, self).__init__(*args, **kwargs)