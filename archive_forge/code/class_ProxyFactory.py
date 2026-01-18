from twisted.internet import protocol
from twisted.python import log
class ProxyFactory(protocol.Factory):
    """
    Factory for port forwarder.
    """
    protocol = ProxyServer

    def __init__(self, host, port):
        self.host = host
        self.port = port