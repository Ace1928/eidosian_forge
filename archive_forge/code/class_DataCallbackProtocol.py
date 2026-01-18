import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class DataCallbackProtocol(protocol.Protocol):

    def dataReceived(self, data):
        d, self.factory.onData = (self.factory.onData, None)
        if d is not None:
            d.callback(data)

    def connectionLost(self, reason):
        d, self.factory.onLost = (self.factory.onLost, None)
        if d is not None:
            d.errback(reason)