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
@implementer(IOpenSSLTrustRoot)
class OpenSSLCertificateAuthorities:
    """
    Trust an explicitly specified set of certificates, represented by a list of
    L{OpenSSL.crypto.X509} objects.
    """

    def __init__(self, caCerts):
        """
        @param caCerts: The certificate authorities to trust when using this
            object as a C{trustRoot} for L{OpenSSLCertificateOptions}.
        @type caCerts: L{list} of L{OpenSSL.crypto.X509}
        """
        self._caCerts = caCerts

    def _addCACertsToContext(self, context):
        store = context.get_cert_store()
        for cert in self._caCerts:
            store.add_cert(cert)