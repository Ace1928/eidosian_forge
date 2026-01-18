import struct
import time
from zope.interface import implementer
from twisted.internet import interfaces, protocol
@implementer(interfaces.IProducer)
class Chargen(protocol.Protocol):
    """
    Generate repeating noise (RFC 864).
    """
    noise = b'@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ !"#$%&?'

    def connectionMade(self):
        self.transport.registerProducer(self, 0)

    def resumeProducing(self):
        self.transport.write(self.noise)

    def pauseProducing(self):
        pass

    def stopProducing(self):
        pass