from __future__ import annotations
import collections
import contextlib
import itertools
import typing
from contextlib import contextmanager
from cryptography import utils, x509
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.backends.openssl import aead
from cryptography.hazmat.backends.openssl.ciphers import _CipherContext
from cryptography.hazmat.backends.openssl.cmac import _CMACContext
from cryptography.hazmat.backends.openssl.ec import (
from cryptography.hazmat.backends.openssl.rsa import (
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.bindings.openssl import binding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives._asymmetric import AsymmetricPadding
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.ciphers.algorithms import (
from cryptography.hazmat.primitives.ciphers.modes import (
from cryptography.hazmat.primitives.serialization import ssh
from cryptography.hazmat.primitives.serialization.pkcs12 import (
def _load_pkcs7_certificates(self, p7) -> typing.List[x509.Certificate]:
    nid = self._lib.OBJ_obj2nid(p7.type)
    self.openssl_assert(nid != self._lib.NID_undef)
    if nid != self._lib.NID_pkcs7_signed:
        raise UnsupportedAlgorithm('Only basic signed structures are currently supported. NID for this data was {}'.format(nid), _Reasons.UNSUPPORTED_SERIALIZATION)
    certs: list[x509.Certificate] = []
    if p7.d.sign == self._ffi.NULL:
        return certs
    sk_x509 = p7.d.sign.cert
    num = self._lib.sk_X509_num(sk_x509)
    for i in range(num):
        x509 = self._lib.sk_X509_value(sk_x509, i)
        self.openssl_assert(x509 != self._ffi.NULL)
        cert = self._ossl2cert(x509)
        certs.append(cert)
    return certs