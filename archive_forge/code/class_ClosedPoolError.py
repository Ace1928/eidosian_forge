from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class ClosedPoolError(PoolError):
    """Raised when a request enters a pool after the pool has been closed."""
    pass