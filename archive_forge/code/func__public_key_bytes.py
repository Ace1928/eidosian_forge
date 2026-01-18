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
def _public_key_bytes(self, encoding: serialization.Encoding, format: serialization.PublicFormat, key, evp_pkey, cdata) -> bytes:
    if not isinstance(encoding, serialization.Encoding):
        raise TypeError('encoding must be an item from the Encoding enum')
    if not isinstance(format, serialization.PublicFormat):
        raise TypeError('format must be an item from the PublicFormat enum')
    if format is serialization.PublicFormat.SubjectPublicKeyInfo:
        if encoding is serialization.Encoding.PEM:
            write_bio = self._lib.PEM_write_bio_PUBKEY
        elif encoding is serialization.Encoding.DER:
            write_bio = self._lib.i2d_PUBKEY_bio
        else:
            raise ValueError('SubjectPublicKeyInfo works only with PEM or DER encoding')
        return self._bio_func_output(write_bio, evp_pkey)
    if format is serialization.PublicFormat.PKCS1:
        key_type = self._lib.EVP_PKEY_id(evp_pkey)
        if key_type != self._lib.EVP_PKEY_RSA:
            raise ValueError('PKCS1 format is supported only for RSA keys')
        if encoding is serialization.Encoding.PEM:
            write_bio = self._lib.PEM_write_bio_RSAPublicKey
        elif encoding is serialization.Encoding.DER:
            write_bio = self._lib.i2d_RSAPublicKey_bio
        else:
            raise ValueError('PKCS1 works only with PEM or DER encoding')
        return self._bio_func_output(write_bio, cdata)
    if format is serialization.PublicFormat.OpenSSH:
        if encoding is serialization.Encoding.OpenSSH:
            return ssh.serialize_ssh_public_key(key)
        raise ValueError('OpenSSH format must be used with OpenSSH encoding')
    raise ValueError('format is invalid with this key')