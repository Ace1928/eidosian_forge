from collections import deque
from contextlib import contextmanager
import sys
import time
from eventlet.pools import Pool
from eventlet import timeout
from eventlet import hubs
from eventlet.hubs.timer import Timer
from eventlet.greenthread import GreenThread
class DatabaseConnector:
    """
    This is an object which will maintain a collection of database
    connection pools on a per-host basis.
    """

    def __init__(self, module, credentials, conn_pool=None, *args, **kwargs):
        """constructor
        *module*
            Database module to use.
        *credentials*
            Mapping of hostname to connect arguments (e.g. username and password)
        """
        assert module
        self._conn_pool_class = conn_pool
        if self._conn_pool_class is None:
            self._conn_pool_class = ConnectionPool
        self._module = module
        self._args = args
        self._kwargs = kwargs
        self._credentials = credentials
        self._databases = {}

    def credentials_for(self, host):
        if host in self._credentials:
            return self._credentials[host]
        else:
            return self._credentials.get('default', None)

    def get(self, host, dbname):
        """Returns a ConnectionPool to the target host and schema.
        """
        key = (host, dbname)
        if key not in self._databases:
            new_kwargs = self._kwargs.copy()
            new_kwargs['db'] = dbname
            new_kwargs['host'] = host
            new_kwargs.update(self.credentials_for(host))
            dbpool = self._conn_pool_class(self._module, *self._args, **new_kwargs)
            self._databases[key] = dbpool
        return self._databases[key]