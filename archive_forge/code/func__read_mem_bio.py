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
def _read_mem_bio(self, bio) -> bytes:
    """
        Reads a memory BIO. This only works on memory BIOs.
        """
    buf = self._ffi.new('char **')
    buf_len = self._lib.BIO_get_mem_data(bio, buf)
    self.openssl_assert(buf_len > 0)
    self.openssl_assert(buf[0] != self._ffi.NULL)
    bio_data = self._ffi.buffer(buf[0], buf_len)[:]
    return bio_data