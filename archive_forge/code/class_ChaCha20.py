from __future__ import annotations
from cryptography import utils
from cryptography.hazmat.primitives.ciphers import (
class ChaCha20(CipherAlgorithm):
    name = 'ChaCha20'
    key_sizes = frozenset([256])

    def __init__(self, key: bytes, nonce: bytes):
        self.key = _verify_key_size(self, key)
        utils._check_byteslike('nonce', nonce)
        if len(nonce) != 16:
            raise ValueError('nonce must be 128-bits (16 bytes)')
        self._nonce = nonce

    @property
    def nonce(self) -> bytes:
        return self._nonce

    @property
    def key_size(self) -> int:
        return len(self.key) * 8