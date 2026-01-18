from __future__ import annotations
import warnings
from binascii import hexlify
from functools import lru_cache
from hashlib import md5
from typing import Dict
from zope.interface import Interface, implementer
from OpenSSL import SSL, crypto
from OpenSSL._util import lib as pyOpenSSLlib
import attr
from constantly import FlagConstant, Flags, NamedConstant, Names
from incremental import Version
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.error import CertificateError, VerifyError
from twisted.internet.interfaces import (
from twisted.python import log, util
from twisted.python.compat import nativeString
from twisted.python.deprecate import _mutuallyExclusiveArguments, deprecated
from twisted.python.failure import Failure
from twisted.python.randbytes import secureRandom
from ._idna import _idnaBytes
class _ChooseDiffieHellmanEllipticCurve:
    """
    Chooses the best elliptic curve for Elliptic Curve Diffie-Hellman
    key exchange, and provides a C{configureECDHCurve} method to set
    the curve, when appropriate, on a new L{OpenSSL.SSL.Context}.

    The C{configureECDHCurve} method will be set to one of the
    following based on the provided OpenSSL version and configuration:

        - L{_configureOpenSSL110}

        - L{_configureOpenSSL102}

        - L{_configureOpenSSL101}

        - L{_configureOpenSSL101NoCurves}.

    @param openSSLVersion: The OpenSSL version number.
    @type openSSLVersion: L{int}

    @see: L{OpenSSL.SSL.OPENSSL_VERSION_NUMBER}

    @param openSSLlib: The OpenSSL C{cffi} library module.
    @param openSSLcrypto: The OpenSSL L{crypto} module.

    @see: L{crypto}
    """

    def __init__(self, openSSLVersion, openSSLlib, openSSLcrypto):
        self._openSSLlib = openSSLlib
        self._openSSLcrypto = openSSLcrypto
        if openSSLVersion >= 269484032:
            self.configureECDHCurve = self._configureOpenSSL110
        elif openSSLVersion >= 268443648:
            self.configureECDHCurve = self._configureOpenSSL102
        else:
            try:
                self._ecCurve = openSSLcrypto.get_elliptic_curve(_defaultCurveName)
            except ValueError:
                self.configureECDHCurve = self._configureOpenSSL101NoCurves
            else:
                self.configureECDHCurve = self._configureOpenSSL101

    def _configureOpenSSL110(self, ctx):
        """
        OpenSSL 1.1.0 Contexts are preconfigured with an optimal set
        of ECDH curves.  This method does nothing.

        @param ctx: L{OpenSSL.SSL.Context}
        """

    def _configureOpenSSL102(self, ctx):
        """
        Have the context automatically choose elliptic curves for
        ECDH.  Run on OpenSSL 1.0.2 and OpenSSL 1.1.0+, but only has
        an effect on OpenSSL 1.0.2.

        @param ctx: The context which .
        @type ctx: L{OpenSSL.SSL.Context}
        """
        ctxPtr = ctx._context
        try:
            self._openSSLlib.SSL_CTX_set_ecdh_auto(ctxPtr, True)
        except BaseException:
            pass

    def _configureOpenSSL101(self, ctx):
        """
        Set the default elliptic curve for ECDH on the context.  Only
        run on OpenSSL 1.0.1.

        @param ctx: The context on which to set the ECDH curve.
        @type ctx: L{OpenSSL.SSL.Context}
        """
        try:
            ctx.set_tmp_ecdh(self._ecCurve)
        except BaseException:
            pass

    def _configureOpenSSL101NoCurves(self, ctx):
        """
        No elliptic curves are available on OpenSSL 1.0.1. We can't
        set anything, so do nothing.

        @param ctx: The context on which to set the ECDH curve.
        @type ctx: L{OpenSSL.SSL.Context}
        """