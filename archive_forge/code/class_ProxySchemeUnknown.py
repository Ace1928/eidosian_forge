from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class ProxySchemeUnknown(AssertionError, URLSchemeUnknown):
    """ProxyManager does not support the supplied scheme"""

    def __init__(self, scheme):
        if scheme == 'localhost':
            scheme = None
        if scheme is None:
            message = 'Proxy URL had no scheme, should start with http:// or https://'
        else:
            message = 'Proxy URL had unsupported scheme %s, should use http:// or https://' % scheme
        super(ProxySchemeUnknown, self).__init__(message)