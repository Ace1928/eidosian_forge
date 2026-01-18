from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class NewConnectionError(ConnectTimeoutError, PoolError):
    """Raised when we fail to establish a new connection. Usually ECONNREFUSED."""
    pass