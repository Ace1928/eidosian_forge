import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
class SingleLineServerProtocol(protocol.Protocol):
    """
    A protocol that sends a single line of data at C{connectionMade}.
    """

    def connectionMade(self):
        self.transport.write(b'+OK <some crap>\r\n')
        self.transport.getPeerCertificate()