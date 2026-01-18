import collections
import contextlib
import itertools
import queue
import threading
import time
import memcache
from oslo_log import log
from oslo_cache._i18n import _
from oslo_cache import exception
class MemcacheClientPool(ConnectionPool):

    def __init__(self, urls, arguments, **kwargs):
        ConnectionPool.__init__(self, **kwargs)
        self.urls = urls
        self._arguments = {'dead_retry': arguments.get('dead_retry', 5 * 60), 'socket_timeout': arguments.get('socket_timeout', 3.0), 'server_max_value_length': arguments.get('server_max_value_length'), 'flush_on_reconnect': arguments.get('pool_flush_on_reconnect', False)}
        self._hosts_deaduntil = [0] * len(urls)

    def _create_connection(self):
        return _MemcacheClient(self.urls, **self._arguments)

    def _destroy_connection(self, conn):
        conn.disconnect_all()

    def _get(self):
        conn = ConnectionPool._get(self)
        try:
            now = time.time()
            for deaduntil, host in zip(self._hosts_deaduntil, conn.servers):
                if deaduntil > now and host.deaduntil <= now:
                    host.mark_dead('propagating death mark from the pool')
                host.deaduntil = deaduntil
        except Exception:
            ConnectionPool._put(self, conn)
            raise
        return conn

    def _put(self, conn):
        try:
            now = time.time()
            for i, host in zip(itertools.count(), conn.servers):
                deaduntil = self._hosts_deaduntil[i]
                if deaduntil <= now:
                    if host.deaduntil > now:
                        self._hosts_deaduntil[i] = host.deaduntil
                        self._debug_logger('Marked host %s dead until %s', self.urls[i], host.deaduntil)
                    else:
                        self._hosts_deaduntil[i] = 0
        finally:
            ConnectionPool._put(self, conn)