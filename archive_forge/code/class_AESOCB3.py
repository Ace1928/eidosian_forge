from __future__ import annotations
import os
import typing
from cryptography import exceptions, utils
from cryptography.hazmat.backends.openssl import aead
from cryptography.hazmat.backends.openssl.backend import backend
from cryptography.hazmat.bindings._rust import FixedPool
class AESOCB3:
    _MAX_SIZE = 2 ** 31 - 1

    def __init__(self, key: bytes):
        utils._check_byteslike('key', key)
        if len(key) not in (16, 24, 32):
            raise ValueError('AESOCB3 key must be 128, 192, or 256 bits.')
        self._key = key
        if not backend.aead_cipher_supported(self):
            raise exceptions.UnsupportedAlgorithm('OCB3 is not supported by this version of OpenSSL', exceptions._Reasons.UNSUPPORTED_CIPHER)

    @classmethod
    def generate_key(cls, bit_length: int) -> bytes:
        if not isinstance(bit_length, int):
            raise TypeError('bit_length must be an integer')
        if bit_length not in (128, 192, 256):
            raise ValueError('bit_length must be 128, 192, or 256')
        return os.urandom(bit_length // 8)

    def encrypt(self, nonce: bytes, data: bytes, associated_data: typing.Optional[bytes]) -> bytes:
        if associated_data is None:
            associated_data = b''
        if len(data) > self._MAX_SIZE or len(associated_data) > self._MAX_SIZE:
            raise OverflowError('Data or associated data too long. Max 2**31 - 1 bytes')
        self._check_params(nonce, data, associated_data)
        return aead._encrypt(backend, self, nonce, data, [associated_data], 16)

    def decrypt(self, nonce: bytes, data: bytes, associated_data: typing.Optional[bytes]) -> bytes:
        if associated_data is None:
            associated_data = b''
        self._check_params(nonce, data, associated_data)
        return aead._decrypt(backend, self, nonce, data, [associated_data], 16)

    def _check_params(self, nonce: bytes, data: bytes, associated_data: bytes) -> None:
        utils._check_byteslike('nonce', nonce)
        utils._check_byteslike('data', data)
        utils._check_byteslike('associated_data', associated_data)
        if len(nonce) < 12 or len(nonce) > 15:
            raise ValueError('Nonce must be between 12 and 15 bytes')