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
class FakeCryptoState:
    """
    State for L{FakeCrypto}

    @param getEllipticCurveRaises: What
        L{FakeCrypto.get_elliptic_curve} should raise; L{None} and it
        won't raise anything

    @param getEllipticCurveReturns: What
        L{FakeCrypto.get_elliptic_curve} should return.

    @ivar getEllipticCurveCalls: The arguments with which
        L{FakeCrypto.get_elliptic_curve} has been called.
    @type getEllipticCurveCalls: L{list}
    """
    __slots__ = ('getEllipticCurveRaises', 'getEllipticCurveReturns', 'getEllipticCurveCalls')

    def __init__(self, getEllipticCurveRaises, getEllipticCurveReturns):
        self.getEllipticCurveRaises = getEllipticCurveRaises
        self.getEllipticCurveReturns = getEllipticCurveReturns
        self.getEllipticCurveCalls = []