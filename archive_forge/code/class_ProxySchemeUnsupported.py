from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class ProxySchemeUnsupported(ValueError):
    """Fetching HTTPS resources through HTTPS proxies is unsupported"""
    pass