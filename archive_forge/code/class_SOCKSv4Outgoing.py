import socket
import string
import struct
import time
from twisted.internet import defer, protocol, reactor
from twisted.python import log
class SOCKSv4Outgoing(protocol.Protocol):

    def __init__(self, socks):
        self.socks = socks

    def connectionMade(self):
        peer = self.transport.getPeer()
        self.socks.makeReply(90, 0, port=peer.port, ip=peer.host)
        self.socks.otherConn = self

    def connectionLost(self, reason):
        self.socks.transport.loseConnection()

    def dataReceived(self, data):
        self.socks.write(data)

    def write(self, data):
        self.socks.log(self, data)
        self.transport.write(data)