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
@lru_cache(maxsize=128)
def _selectCiphers(wantedCiphers, availableCiphers):
    """
    Caclulate the acceptable list of ciphers from the ciphers we want and the
    ciphers we have support for.

    @param wantedCiphers: The ciphers we want to use.
    @type wantedCiphers: L{tuple} of L{OpenSSLCipher}

    @param availableCiphers: The ciphers we have available to use.
    @type availableCiphers: L{tuple} of L{OpenSSLCipher}

    @rtype: L{tuple} of L{OpenSSLCipher}
    """
    return tuple((cipher for cipher in wantedCiphers if cipher in availableCiphers))