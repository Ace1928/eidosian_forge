import inspect
import sys
from magnumclient.i18n import _
class NotAcceptable(HTTPClientError):
    """HTTP 406 - Not Acceptable.

    The requested resource is only capable of generating content not
    acceptable according to the Accept headers sent in the request.
    """
    http_status = 406
    message = _('Not Acceptable')