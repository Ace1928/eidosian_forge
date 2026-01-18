from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class LocationParseError(LocationValueError):
    """Raised when get_host or similar fails to parse the URL input."""

    def __init__(self, location):
        message = 'Failed to parse: %s' % location
        HTTPError.__init__(self, message)
        self.location = location