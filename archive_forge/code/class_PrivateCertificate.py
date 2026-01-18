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
class PrivateCertificate(Certificate):
    """
    An x509 certificate and private key.
    """

    def __repr__(self) -> str:
        return Certificate.__repr__(self) + ' with ' + repr(self.privateKey)

    def _setPrivateKey(self, privateKey):
        if not privateKey.matches(self.getPublicKey()):
            raise VerifyError('Certificate public and private keys do not match.')
        self.privateKey = privateKey
        return self

    def newCertificate(self, newCertData, format=crypto.FILETYPE_ASN1):
        """
        Create a new L{PrivateCertificate} from the given certificate data and
        this instance's private key.
        """
        return self.load(newCertData, self.privateKey, format)

    @classmethod
    def load(Class, data, privateKey, format=crypto.FILETYPE_ASN1):
        return Class._load(data, format)._setPrivateKey(privateKey)

    def inspect(self):
        return '\n'.join([Certificate._inspect(self), self.privateKey.inspect()])

    def dumpPEM(self):
        """
        Dump both public and private parts of a private certificate to
        PEM-format data.
        """
        return self.dump(crypto.FILETYPE_PEM) + self.privateKey.dump(crypto.FILETYPE_PEM)

    @classmethod
    def loadPEM(Class, data):
        """
        Load both private and public parts of a private certificate from a
        chunk of PEM-format data.
        """
        return Class.load(data, KeyPair.load(data, crypto.FILETYPE_PEM), crypto.FILETYPE_PEM)

    @classmethod
    def fromCertificateAndKeyPair(Class, certificateInstance, privateKey):
        privcert = Class(certificateInstance.original)
        return privcert._setPrivateKey(privateKey)

    def options(self, *authorities):
        """
        Create a context factory using this L{PrivateCertificate}'s certificate
        and private key.

        @param authorities: A list of L{Certificate} object

        @return: A context factory.
        @rtype: L{CertificateOptions <twisted.internet.ssl.CertificateOptions>}
        """
        options = dict(privateKey=self.privateKey.original, certificate=self.original)
        if authorities:
            options.update(dict(trustRoot=OpenSSLCertificateAuthorities([auth.original for auth in authorities])))
        return OpenSSLCertificateOptions(**options)

    def certificateRequest(self, format=crypto.FILETYPE_ASN1, digestAlgorithm='sha256'):
        return self.privateKey.certificateRequest(self.getSubject(), format, digestAlgorithm)

    def signCertificateRequest(self, requestData, verifyDNCallback, serialNumber, requestFormat=crypto.FILETYPE_ASN1, certificateFormat=crypto.FILETYPE_ASN1):
        issuer = self.getSubject()
        return self.privateKey.signCertificateRequest(issuer, requestData, verifyDNCallback, serialNumber, requestFormat, certificateFormat)

    def signRequestObject(self, certificateRequest, serialNumber, secondsToExpiry=60 * 60 * 24 * 365, digestAlgorithm='sha256'):
        return self.privateKey.signRequestObject(self.getSubject(), certificateRequest, serialNumber, secondsToExpiry, digestAlgorithm)