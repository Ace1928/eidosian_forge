import inspect
import sys
from magnumclient.i18n import _
class ProxyAuthenticationRequired(HTTPClientError):
    """HTTP 407 - Proxy Authentication Required.

    The client must first authenticate itself with the proxy.
    """
    http_status = 407
    message = _('Proxy Authentication Required')