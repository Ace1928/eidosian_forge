import time
import hashlib
from libcloud.utils.py3 import b
from libcloud.common.base import ConnectionKey
from libcloud.common.xmlrpc import XMLRPCResponse, XMLRPCConnection
class GandiConnection(XMLRPCConnection, ConnectionKey):
    """
    Connection class for the Gandi driver
    """
    responseCls = GandiResponse
    host = 'rpc.gandi.net'
    endpoint = '/xmlrpc/'

    def __init__(self, key, secure=True, timeout=None, retry_delay=None, backoff=None, proxy_url=None):
        super().__init__(key=key, secure=secure, timeout=timeout, retry_delay=retry_delay, backoff=backoff, proxy_url=proxy_url)
        self.driver = BaseGandiDriver

    def request(self, method, *args):
        args = (self.key,) + args
        return super().request(method, *args)