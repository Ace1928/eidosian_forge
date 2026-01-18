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
class GetCipherByName:

    def __init__(self, fmt: str):
        self._fmt = fmt

    def __call__(self, backend: Backend, cipher: CipherAlgorithm, mode: Mode):
        cipher_name = self._fmt.format(cipher=cipher, mode=mode).lower()
        evp_cipher = backend._lib.EVP_get_cipherbyname(cipher_name.encode('ascii'))
        if evp_cipher == backend._ffi.NULL and backend._lib.Cryptography_HAS_300_EVP_CIPHER:
            evp_cipher = backend._lib.EVP_CIPHER_fetch(backend._ffi.NULL, cipher_name.encode('ascii'), backend._ffi.NULL)
        backend._consume_errors()
        return evp_cipher