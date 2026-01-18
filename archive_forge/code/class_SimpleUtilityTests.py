import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
@skipIf(ipv6Skip, ipv6SkipReason)
class SimpleUtilityTests(TestCase):
    """
    Simple, direct tests for helpers within L{twisted.internet.tcp}.
    """

    def test_resolveNumericHost(self):
        """
        L{_resolveIPv6} raises a L{socket.gaierror} (L{socket.EAI_NONAME}) when
        invoked with a non-numeric host.  (In other words, it is passing
        L{socket.AI_NUMERICHOST} to L{socket.getaddrinfo} and will not
        accidentally block if it receives bad input.)
        """
        err = self.assertRaises(socket.gaierror, _resolveIPv6, 'localhost', 1)
        self.assertEqual(err.args[0], socket.EAI_NONAME)

    @skipIf(platform.isWindows(), 'The AI_NUMERICSERV flag is not supported by Microsoft providers.')
    def test_resolveNumericService(self):
        """
        L{_resolveIPv6} raises a L{socket.gaierror} (L{socket.EAI_NONAME}) when
        invoked with a non-numeric port.  (In other words, it is passing
        L{socket.AI_NUMERICSERV} to L{socket.getaddrinfo} and will not
        accidentally block if it receives bad input.)
        """
        err = self.assertRaises(socket.gaierror, _resolveIPv6, '::1', 'http')
        self.assertEqual(err.args[0], socket.EAI_NONAME)

    def test_resolveIPv6(self):
        """
        L{_resolveIPv6} discovers the flow info and scope ID of an IPv6
        address.
        """
        result = _resolveIPv6('::1', 2)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[2], int)
        self.assertIsInstance(result[3], int)
        self.assertEqual(result[:2], ('::1', 2))