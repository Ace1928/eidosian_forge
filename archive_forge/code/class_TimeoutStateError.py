from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class TimeoutStateError(HTTPError):
    """Raised when passing an invalid state to a timeout"""
    pass