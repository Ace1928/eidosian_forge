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
def SSL_CTX_set_ecdh_auto(self, ctx, value):
    """
        Record the context and value under in the C{_state} instance
        variable.

        @see: L{FakeLibState}

        @param ctx: An SSL context.
        @type ctx: L{OpenSSL.SSL.Context}

        @param value: A boolean value
        @type value: L{bool}
        """
    self._state.ecdhContexts.append(ctx)
    self._state.ecdhValues.append(value)
    if self._state.setECDHAutoRaises is not None:
        raise self._state.setECDHAutoRaises