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
def _private_key_bytes(self, encoding: serialization.Encoding, format: serialization.PrivateFormat, encryption_algorithm: serialization.KeySerializationEncryption, key, evp_pkey, cdata) -> bytes:
    if not isinstance(encoding, serialization.Encoding):
        raise TypeError('encoding must be an item from the Encoding enum')
    if not isinstance(format, serialization.PrivateFormat):
        raise TypeError('format must be an item from the PrivateFormat enum')
    if not isinstance(encryption_algorithm, serialization.KeySerializationEncryption):
        raise TypeError('Encryption algorithm must be a KeySerializationEncryption instance')
    if isinstance(encryption_algorithm, serialization.NoEncryption):
        password = b''
    elif isinstance(encryption_algorithm, serialization.BestAvailableEncryption):
        password = encryption_algorithm.password
        if len(password) > 1023:
            raise ValueError('Passwords longer than 1023 bytes are not supported by this backend')
    elif isinstance(encryption_algorithm, serialization._KeySerializationEncryption) and encryption_algorithm._format is format is serialization.PrivateFormat.OpenSSH:
        password = encryption_algorithm.password
    else:
        raise ValueError('Unsupported encryption type')
    if format is serialization.PrivateFormat.PKCS8:
        if encoding is serialization.Encoding.PEM:
            write_bio = self._lib.PEM_write_bio_PKCS8PrivateKey
        elif encoding is serialization.Encoding.DER:
            write_bio = self._lib.i2d_PKCS8PrivateKey_bio
        else:
            raise ValueError('Unsupported encoding for PKCS8')
        return self._private_key_bytes_via_bio(write_bio, evp_pkey, password)
    if format is serialization.PrivateFormat.TraditionalOpenSSL:
        if self._fips_enabled and (not isinstance(encryption_algorithm, serialization.NoEncryption)):
            raise ValueError('Encrypted traditional OpenSSL format is not supported in FIPS mode.')
        key_type = self._lib.EVP_PKEY_id(evp_pkey)
        if encoding is serialization.Encoding.PEM:
            if key_type == self._lib.EVP_PKEY_RSA:
                write_bio = self._lib.PEM_write_bio_RSAPrivateKey
            else:
                assert key_type == self._lib.EVP_PKEY_EC
                write_bio = self._lib.PEM_write_bio_ECPrivateKey
            return self._private_key_bytes_via_bio(write_bio, cdata, password)
        if encoding is serialization.Encoding.DER:
            if password:
                raise ValueError('Encryption is not supported for DER encoded traditional OpenSSL keys')
            if key_type == self._lib.EVP_PKEY_RSA:
                write_bio = self._lib.i2d_RSAPrivateKey_bio
            else:
                assert key_type == self._lib.EVP_PKEY_EC
                write_bio = self._lib.i2d_ECPrivateKey_bio
            return self._bio_func_output(write_bio, cdata)
        raise ValueError('Unsupported encoding for TraditionalOpenSSL')
    if format is serialization.PrivateFormat.OpenSSH:
        if encoding is serialization.Encoding.PEM:
            return ssh._serialize_ssh_private_key(key, password, encryption_algorithm)
        raise ValueError('OpenSSH private key format can only be used with PEM encoding')
    raise ValueError('format is invalid with this key')