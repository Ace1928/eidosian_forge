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
class RememberingWrapper(protocol.ClientFactory):
    """
            Simple wrapper factory which records the addresses which are
            passed to its L{buildProtocol} method and delegates actual
            protocol creation to another factory.

            @ivar addresses: A list of the objects passed to buildProtocol.
            @ivar factory: The wrapped factory to which protocol creation is
                delegated.
            """

    def __init__(self, factory):
        self.addresses = []
        self.factory = factory

    def buildProtocol(self, addr):
        """
                Append the given address to C{self.addresses} and forward
                the call to C{self.factory}.
                """
        self.addresses.append(addr)
        return self.factory.buildProtocol(addr)