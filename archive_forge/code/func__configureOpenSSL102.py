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