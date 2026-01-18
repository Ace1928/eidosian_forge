from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class BLAKE2s(HashAlgorithm):
    name = 'blake2s'
    block_size = 64
    _max_digest_size = 32
    _min_digest_size = 1

    def __init__(self, digest_size: int):
        if digest_size != 32:
            raise ValueError('Digest size must be 32')
        self._digest_size = digest_size

    @property
    def digest_size(self) -> int:
        return self._digest_size