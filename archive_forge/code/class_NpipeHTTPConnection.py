import queue
import requests.adapters
from docker.transport.basehttpadapter import BaseHTTPAdapter
from .. import constants
from .npipesocket import NpipeSocket
import urllib3
import urllib3.connection
class NpipeHTTPConnection(urllib3.connection.HTTPConnection):

    def __init__(self, npipe_path, timeout=60):
        super().__init__('localhost', timeout=timeout)
        self.npipe_path = npipe_path
        self.timeout = timeout

    def connect(self):
        sock = NpipeSocket()
        sock.settimeout(self.timeout)
        sock.connect(self.npipe_path)
        self.sock = sock