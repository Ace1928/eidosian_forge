import socket
from gc import collect
from typing import Optional
from weakref import ref
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.interfaces import IConnector, IReactorFDSet
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.reactormixins import needsRunningReactor
from twisted.python import context, log
from twisted.python.failure import Failure
from twisted.python.log import ILogContext, err, msg
from twisted.python.runtime import platform
from twisted.test.test_tcp import ClosingProtocol
from twisted.trial.unittest import SkipTest
class _SimplePullProducer:
    """
    A pull producer which writes one byte whenever it is resumed.  For use by
    C{test_unregisterProducerAfterDisconnect}.
    """

    def __init__(self, consumer):
        self.consumer = consumer

    def stopProducing(self):
        pass

    def resumeProducing(self):
        log.msg('Producer.resumeProducing')
        self.consumer.write(b'x')