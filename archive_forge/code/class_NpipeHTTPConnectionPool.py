import queue
import requests.adapters
from docker.transport.basehttpadapter import BaseHTTPAdapter
from .. import constants
from .npipesocket import NpipeSocket
import urllib3
import urllib3.connection
class NpipeHTTPConnectionPool(urllib3.connectionpool.HTTPConnectionPool):

    def __init__(self, npipe_path, timeout=60, maxsize=10):
        super().__init__('localhost', timeout=timeout, maxsize=maxsize)
        self.npipe_path = npipe_path
        self.timeout = timeout

    def _new_conn(self):
        return NpipeHTTPConnection(self.npipe_path, self.timeout)

    def _get_conn(self, timeout):
        conn = None
        try:
            conn = self.pool.get(block=self.block, timeout=timeout)
        except AttributeError:
            raise urllib3.exceptions.ClosedPoolError(self, 'Pool is closed.')
        except queue.Empty:
            if self.block:
                raise urllib3.exceptions.EmptyPoolError(self, 'Pool reached maximum size and no more connections are allowed.')
        return conn or self._new_conn()