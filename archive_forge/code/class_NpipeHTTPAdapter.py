import queue
import requests.adapters
from docker.transport.basehttpadapter import BaseHTTPAdapter
from .. import constants
from .npipesocket import NpipeSocket
import urllib3
import urllib3.connection
class NpipeHTTPAdapter(BaseHTTPAdapter):
    __attrs__ = requests.adapters.HTTPAdapter.__attrs__ + ['npipe_path', 'pools', 'timeout', 'max_pool_size']

    def __init__(self, base_url, timeout=60, pool_connections=constants.DEFAULT_NUM_POOLS, max_pool_size=constants.DEFAULT_MAX_POOL_SIZE):
        self.npipe_path = base_url.replace('npipe://', '')
        self.timeout = timeout
        self.max_pool_size = max_pool_size
        self.pools = RecentlyUsedContainer(pool_connections, dispose_func=lambda p: p.close())
        super().__init__()

    def get_connection(self, url, proxies=None):
        with self.pools.lock:
            pool = self.pools.get(url)
            if pool:
                return pool
            pool = NpipeHTTPConnectionPool(self.npipe_path, self.timeout, maxsize=self.max_pool_size)
            self.pools[url] = pool
        return pool

    def request_url(self, request, proxies):
        return request.path_url