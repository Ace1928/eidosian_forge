import requests.adapters
import socket
from docker.transport.basehttpadapter import BaseHTTPAdapter
from .. import constants
import urllib3
import urllib3.connection
class UnixHTTPConnectionPool(urllib3.connectionpool.HTTPConnectionPool):

    def __init__(self, base_url, socket_path, timeout=60, maxsize=10):
        super().__init__('localhost', timeout=timeout, maxsize=maxsize)
        self.base_url = base_url
        self.socket_path = socket_path
        self.timeout = timeout

    def _new_conn(self):
        return UnixHTTPConnection(self.base_url, self.socket_path, self.timeout)