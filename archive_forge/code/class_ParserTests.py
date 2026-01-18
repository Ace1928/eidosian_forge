from errno import EPERM
from socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, AddressFamily, gaierror
from types import FunctionType
from unicodedata import normalize
from unittest import skipIf
from zope.interface import implementer, providedBy, provider
from zope.interface.interface import InterfaceClass
from zope.interface.verify import verifyClass, verifyObject
from twisted import plugins
from twisted.internet import (
from twisted.internet.abstract import isIPv6Address
from twisted.internet.address import (
from twisted.internet.endpoints import StandardErrorBehavior
from twisted.internet.error import ConnectingCancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol
from twisted.internet.stdio import PipeAddress
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import ILogObserver, globalLogPublisher
from twisted.plugin import getPlugins
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.systemd import ListenFDs
from twisted.test.iosim import connectableEndpoint, connectedServerAndClient
from twisted.trial import unittest
class ParserTests(unittest.TestCase):
    """
    Tests for L{endpoints._parseServer}, the low-level parsing logic.
    """
    f = 'Factory'

    def parse(self, *a, **kw):
        """
        Provide a hook for test_strports to substitute the deprecated API.
        """
        return endpoints._parseServer(*a, **kw)

    def test_simpleTCP(self):
        """
        Simple strings with a 'tcp:' prefix should be parsed as TCP.
        """
        self.assertEqual(self.parse('tcp:80', self.f), ('TCP', (80, self.f), {'interface': '', 'backlog': 50}))

    def test_interfaceTCP(self):
        """
        TCP port descriptions parse their 'interface' argument as a string.
        """
        self.assertEqual(self.parse('tcp:80:interface=127.0.0.1', self.f), ('TCP', (80, self.f), {'interface': '127.0.0.1', 'backlog': 50}))

    def test_backlogTCP(self):
        """
        TCP port descriptions parse their 'backlog' argument as an integer.
        """
        self.assertEqual(self.parse('tcp:80:backlog=6', self.f), ('TCP', (80, self.f), {'interface': '', 'backlog': 6}))

    def test_simpleUNIX(self):
        """
        L{endpoints._parseServer} returns a C{'UNIX'} port description with
        defaults for C{'mode'}, C{'backlog'}, and C{'wantPID'} when passed a
        string with the C{'unix:'} prefix and no other parameter values.
        """
        self.assertEqual(self.parse('unix:/var/run/finger', self.f), ('UNIX', ('/var/run/finger', self.f), {'mode': 438, 'backlog': 50, 'wantPID': True}))

    def test_modeUNIX(self):
        """
        C{mode} can be set by including C{"mode=<some integer>"}.
        """
        self.assertEqual(self.parse('unix:/var/run/finger:mode=0660', self.f), ('UNIX', ('/var/run/finger', self.f), {'mode': 432, 'backlog': 50, 'wantPID': True}))

    def test_wantPIDUNIX(self):
        """
        C{wantPID} can be set to false by included C{"lockfile=0"}.
        """
        self.assertEqual(self.parse('unix:/var/run/finger:lockfile=0', self.f), ('UNIX', ('/var/run/finger', self.f), {'mode': 438, 'backlog': 50, 'wantPID': False}))

    def test_escape(self):
        """
        Backslash can be used to escape colons and backslashes in port
        descriptions.
        """
        self.assertEqual(self.parse('unix:foo\\:bar\\=baz\\:qux\\\\', self.f), ('UNIX', ('foo:bar=baz:qux\\', self.f), {'mode': 438, 'backlog': 50, 'wantPID': True}))

    def test_quoteStringArgument(self):
        """
        L{endpoints.quoteStringArgument} should quote backslashes and colons
        for interpolation into L{endpoints.serverFromString} and
        L{endpoints.clientFactory} arguments.
        """
        self.assertEqual(endpoints.quoteStringArgument('some : stuff \\'), 'some \\: stuff \\\\')

    def test_impliedEscape(self):
        """
        In strports descriptions, '=' in a parameter value does not need to be
        quoted; it will simply be parsed as part of the value.
        """
        self.assertEqual(self.parse('unix:address=foo=bar', self.f), ('UNIX', ('foo=bar', self.f), {'mode': 438, 'backlog': 50, 'wantPID': True}))

    def test_unknownType(self):
        """
        L{strports.parse} raises C{ValueError} when given an unknown endpoint
        type.
        """
        self.assertRaises(ValueError, self.parse, 'bogus-type:nothing', self.f)