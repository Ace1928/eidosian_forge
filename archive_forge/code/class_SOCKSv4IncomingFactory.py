import socket
import string
import struct
import time
from twisted.internet import defer, protocol, reactor
from twisted.python import log
class SOCKSv4IncomingFactory(protocol.Factory):
    """
    A utility class for building protocols for incoming connections.
    """

    def __init__(self, socks, ip):
        self.socks = socks
        self.ip = ip

    def buildProtocol(self, addr):
        if addr[0] == self.ip:
            self.ip = ''
            self.socks.makeReply(90, 0)
            return SOCKSv4Incoming(self.socks)
        elif self.ip == '':
            return None
        else:
            self.socks.makeReply(91, 0)
            self.ip = ''
            return None