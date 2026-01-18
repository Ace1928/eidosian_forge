from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class LocationValueError(ValueError, HTTPError):
    """Raised when there is something wrong with a given URL input."""
    pass