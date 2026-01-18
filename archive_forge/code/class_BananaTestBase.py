import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
class BananaTestBase(TestCase):
    """
    The base for test classes. It defines commonly used things and sets up a
    connection for testing.
    """
    encClass = banana.Banana

    def setUp(self):
        self.io = BytesIO()
        self.enc = self.encClass()
        self.enc.makeConnection(protocol.FileWrapper(self.io))
        selectDialect(self.enc, b'none')
        self.enc.expressionReceived = self.putResult
        self.encode = partial(encode, self.encClass)

    def putResult(self, result):
        """
        Store an expression received by C{self.enc}.

        @param result: The object that was received.
        @type result: Any type supported by Banana.
        """
        self.result = result

    def tearDown(self):
        self.enc.connectionLost(failure.Failure(main.CONNECTION_DONE))
        del self.enc