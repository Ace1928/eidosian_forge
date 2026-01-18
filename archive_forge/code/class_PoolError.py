from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class PoolError(HTTPError):
    """Base exception for errors caused within a pool."""

    def __init__(self, pool, message):
        self.pool = pool
        HTTPError.__init__(self, '%s: %s' % (pool, message))

    def __reduce__(self):
        return (self.__class__, (None, None))