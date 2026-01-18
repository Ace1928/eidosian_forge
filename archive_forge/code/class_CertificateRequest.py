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
class CertificateRequest(CertBase):
    """
    An x509 certificate request.

    Certificate requests are given to certificate authorities to be signed and
    returned resulting in an actual certificate.
    """

    @classmethod
    def load(Class, requestData, requestFormat=crypto.FILETYPE_ASN1):
        req = crypto.load_certificate_request(requestFormat, requestData)
        dn = DistinguishedName()
        dn._copyFrom(req.get_subject())
        if not req.verify(req.get_pubkey()):
            raise VerifyError(f"Can't verify that request for {dn!r} is self-signed.")
        return Class(req)

    def dump(self, format=crypto.FILETYPE_ASN1):
        return crypto.dump_certificate_request(format, self.original)