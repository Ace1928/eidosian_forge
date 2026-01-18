from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
class SaveProtocolServerFactory(pb.PBServerFactory):
    """
    A L{pb.PBServerFactory} that saves the latest connected client in
    C{protocolInstance}.
    """
    protocolInstance = None

    def clientConnectionMade(self, protocol):
        """
        Keep track of the given protocol.
        """
        self.protocolInstance = protocol