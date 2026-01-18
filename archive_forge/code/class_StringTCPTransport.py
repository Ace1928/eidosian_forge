import socket
import struct
from twisted.internet import address, defer
from twisted.internet.error import DNSLookupError
from twisted.protocols import socks
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class StringTCPTransport(proto_helpers.StringTransport):
    stringTCPTransport_closing = False
    peer = None

    def getPeer(self):
        return self.peer

    def getHost(self):
        return address.IPv4Address('TCP', '2.3.4.5', 42)

    def loseConnection(self):
        self.stringTCPTransport_closing = True