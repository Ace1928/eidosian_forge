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
class SelectVerifyImplementationTests(SynchronousTestCase):
    """
    Tests for L{_selectVerifyImplementation}.
    """
    if skipSSL:
        skip = skipSSL

    def test_dependencyMissing(self):
        """
        If I{service_identity} cannot be imported then
        L{_selectVerifyImplementation} returns L{simpleVerifyHostname} and
        L{SimpleVerificationError}.
        """
        with SetAsideModule('service_identity'):
            sys.modules['service_identity'] = None
            result = sslverify._selectVerifyImplementation()
            expected = (sslverify.simpleVerifyHostname, sslverify.simpleVerifyIPAddress, sslverify.SimpleVerificationError)
            self.assertEqual(expected, result)
    test_dependencyMissing.suppress = [util.suppress(message='You do not have a working installation of the service_identity module')]

    def test_dependencyMissingWarning(self):
        """
        If I{service_identity} cannot be imported then
        L{_selectVerifyImplementation} emits a L{UserWarning} advising the user
        of the exact error.
        """
        with SetAsideModule('service_identity'):
            sys.modules['service_identity'] = None
            sslverify._selectVerifyImplementation()
        [warning] = list((warning for warning in self.flushWarnings() if warning['category'] == UserWarning))
        expectedMessage = "You do not have a working installation of the service_identity module: 'import of service_identity halted; None in sys.modules'.  Please install it from <https://pypi.python.org/pypi/service_identity> and make sure all of its dependencies are satisfied.  Without the service_identity module, Twisted can perform only rudimentary TLS client hostname verification.  Many valid certificate/hostname mappings may be rejected."
        self.assertEqual(warning['message'], expectedMessage)
        self.assertEqual(warning['filename'], '')
        self.assertEqual(warning['lineno'], 0)