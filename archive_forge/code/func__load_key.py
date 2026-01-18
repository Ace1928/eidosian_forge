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
def _load_key(self, openssl_read_func, data, password, unsafe_skip_rsa_key_validation) -> PrivateKeyTypes:
    mem_bio = self._bytes_to_bio(data)
    userdata = self._ffi.new('CRYPTOGRAPHY_PASSWORD_DATA *')
    if password is not None:
        utils._check_byteslike('password', password)
        password_ptr = self._ffi.from_buffer(password)
        userdata.password = password_ptr
        userdata.length = len(password)
    evp_pkey = openssl_read_func(mem_bio.bio, self._ffi.NULL, self._ffi.addressof(self._lib._original_lib, 'Cryptography_pem_password_cb'), userdata)
    if evp_pkey == self._ffi.NULL:
        if userdata.error != 0:
            self._consume_errors()
            if userdata.error == -1:
                raise TypeError('Password was not given but private key is encrypted')
            else:
                assert userdata.error == -2
                raise ValueError('Passwords longer than {} bytes are not supported by this backend.'.format(userdata.maxsize - 1))
        else:
            self._handle_key_loading_error()
    evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)
    if password is not None and userdata.called == 0:
        raise TypeError('Password was given but private key is not encrypted.')
    assert password is not None and userdata.called == 1 or password is None
    return self._evp_pkey_to_private_key(evp_pkey, unsafe_skip_rsa_key_validation)