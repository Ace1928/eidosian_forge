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
class FakeContext:
    """
    Introspectable fake of an C{OpenSSL.SSL.Context}.

    Saves call arguments for later introspection.

    Necessary because C{Context} offers poor introspection.  cf. this
    U{pyOpenSSL bug<https://bugs.launchpad.net/pyopenssl/+bug/1173899>}.

    @ivar _method: See C{method} parameter of L{__init__}.

    @ivar _options: L{int} of C{OR}ed values from calls of L{set_options}.

    @ivar _certificate: Set by L{use_certificate}.

    @ivar _privateKey: Set by L{use_privatekey}.

    @ivar _verify: Set by L{set_verify}.

    @ivar _verifyDepth: Set by L{set_verify_depth}.

    @ivar _mode: Set by L{set_mode}.

    @ivar _sessionID: Set by L{set_session_id}.

    @ivar _extraCertChain: Accumulated L{list} of all extra certificates added
        by L{add_extra_chain_cert}.

    @ivar _cipherList: Set by L{set_cipher_list}.

    @ivar _dhFilename: Set by L{load_tmp_dh}.

    @ivar _defaultVerifyPathsSet: Set by L{set_default_verify_paths}

    @ivar _ecCurve: Set by L{set_tmp_ecdh}
    """
    _options = 0

    def __init__(self, method):
        self._method = method
        self._extraCertChain = []
        self._defaultVerifyPathsSet = False
        self._ecCurve = None
        self._sessionCacheMode = SSL.SESS_CACHE_SERVER

    def set_options(self, options):
        self._options |= options

    def use_certificate(self, certificate):
        self._certificate = certificate

    def use_privatekey(self, privateKey):
        self._privateKey = privateKey

    def check_privatekey(self):
        return None

    def set_mode(self, mode):
        """
        Set the mode. See L{SSL.Context.set_mode}.

        @param mode: See L{SSL.Context.set_mode}.
        """
        self._mode = mode

    def set_verify(self, flags, callback=None):
        self._verify = (flags, callback)

    def set_verify_depth(self, depth):
        self._verifyDepth = depth

    def set_session_id(self, sessionIDContext):
        self._sessionIDContext = sessionIDContext

    def set_session_cache_mode(self, cacheMode):
        """
        Set the session cache mode on the context, as per
        L{SSL.Context.set_session_cache_mode}.
        """
        self._sessionCacheMode = cacheMode

    def get_session_cache_mode(self):
        """
        Retrieve the session cache mode from the context, as per
        L{SSL.Context.get_session_cache_mode}.
        """
        return self._sessionCacheMode

    def add_extra_chain_cert(self, cert):
        self._extraCertChain.append(cert)

    def set_cipher_list(self, cipherList):
        self._cipherList = cipherList

    def load_tmp_dh(self, dhfilename):
        self._dhFilename = dhfilename

    def set_default_verify_paths(self):
        """
        Set the default paths for the platform.
        """
        self._defaultVerifyPathsSet = True

    def set_tmp_ecdh(self, curve):
        """
        Set an ECDH curve.  Should only be called by OpenSSL 1.0.1
        code.

        @param curve: See L{OpenSSL.SSL.Context.set_tmp_ecdh}
        """
        self._ecCurve = curve