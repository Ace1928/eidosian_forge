from __future__ import annotations
import os
import typing
from cryptography import exceptions, utils
from cryptography.hazmat.backends.openssl import aead
from cryptography.hazmat.backends.openssl.backend import backend
from cryptography.hazmat.bindings._rust import FixedPool
class ChaCha20Poly1305:
    _MAX_SIZE = 2 ** 31 - 1

    def __init__(self, key: bytes):
        if not backend.aead_cipher_supported(self):
            raise exceptions.UnsupportedAlgorithm('ChaCha20Poly1305 is not supported by this version of OpenSSL', exceptions._Reasons.UNSUPPORTED_CIPHER)
        utils._check_byteslike('key', key)
        if len(key) != 32:
            raise ValueError('ChaCha20Poly1305 key must be 32 bytes.')
        self._key = key
        self._pool = FixedPool(self._create_fn)

    @classmethod
    def generate_key(cls) -> bytes:
        return os.urandom(32)

    def _create_fn(self):
        return aead._aead_create_ctx(backend, self, self._key)

    def encrypt(self, nonce: bytes, data: bytes, associated_data: typing.Optional[bytes]) -> bytes:
        if associated_data is None:
            associated_data = b''
        if len(data) > self._MAX_SIZE or len(associated_data) > self._MAX_SIZE:
            raise OverflowError('Data or associated data too long. Max 2**31 - 1 bytes')
        self._check_params(nonce, data, associated_data)
        with self._pool.acquire() as ctx:
            return aead._encrypt(backend, self, nonce, data, [associated_data], 16, ctx)

    def decrypt(self, nonce: bytes, data: bytes, associated_data: typing.Optional[bytes]) -> bytes:
        if associated_data is None:
            associated_data = b''
        self._check_params(nonce, data, associated_data)
        with self._pool.acquire() as ctx:
            return aead._decrypt(backend, self, nonce, data, [associated_data], 16, ctx)

    def _check_params(self, nonce: bytes, data: bytes, associated_data: bytes) -> None:
        utils._check_byteslike('nonce', nonce)
        utils._check_byteslike('data', data)
        utils._check_byteslike('associated_data', associated_data)
        if len(nonce) != 12:
            raise ValueError('Nonce must be 12 bytes')