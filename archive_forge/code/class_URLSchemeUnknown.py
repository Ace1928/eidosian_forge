from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class URLSchemeUnknown(LocationValueError):
    """Raised when a URL input has an unsupported scheme."""

    def __init__(self, scheme):
        message = 'Not supported URL scheme %s' % scheme
        super(URLSchemeUnknown, self).__init__(message)
        self.scheme = scheme