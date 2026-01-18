from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
class LoopbackUNIXTests(LoopbackTestCaseMixin, unittest.TestCase):
    loopbackFunc = staticmethod(loopback.loopbackUNIX)
    if interfaces.IReactorUNIX(reactor, None) is None:
        skip = 'Current reactor does not support UNIX sockets'