import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def _connect1(results):
    d = defer.Deferred()
    cf1 = MyClientFactory()
    cf1.buildProtocol = self._fireWhenDoneFunc(d, cf1.buildProtocol)
    reactor.connectTCP('127.0.0.1', p.getHost().port, cf1, bindAddress=('127.0.0.1', 0))
    d.addCallback(_conmade, cf1)
    return d