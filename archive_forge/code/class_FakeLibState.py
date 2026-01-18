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
class FakeLibState:
    """
    State for L{FakeLib}

    @param setECDHAutoRaises: An exception
        L{FakeLib.SSL_CTX_set_ecdh_auto} should raise; if L{None},
        nothing is raised.

    @ivar ecdhContexts: A list of SSL contexts with which
        L{FakeLib.SSL_CTX_set_ecdh_auto} was called
    @type ecdhContexts: L{list} of L{OpenSSL.SSL.Context}s

    @ivar ecdhValues: A list of boolean values with which
        L{FakeLib.SSL_CTX_set_ecdh_auto} was called
    @type ecdhValues: L{list} of L{boolean}s
    """
    __slots__ = ('setECDHAutoRaises', 'ecdhContexts', 'ecdhValues')

    def __init__(self, setECDHAutoRaises):
        self.setECDHAutoRaises = setECDHAutoRaises
        self.ecdhContexts = []
        self.ecdhValues = []