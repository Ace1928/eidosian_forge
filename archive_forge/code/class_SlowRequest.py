from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
class SlowRequest:
    """
    L{SlowRequest} is a fake implementation of L{Request} which is easily
    controlled externally (for example, by code in a test method).

    @ivar stopped: A flag indicating whether C{stopWriting} has been called.

    @ivar finished: After C{writeTo} is called, a L{Deferred} which was
        returned by that method.  L{SlowRequest} will never fire this
        L{Deferred}.
    """
    method = b'GET'
    stopped = False
    persistent = False

    def writeTo(self, transport):
        self.finished = Deferred()
        return self.finished

    def stopWriting(self):
        self.stopped = True