from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SHAKE256(HashAlgorithm, ExtendableOutputFunction):
    name = 'shake256'
    block_size = None

    def __init__(self, digest_size: int):
        if not isinstance(digest_size, int):
            raise TypeError('digest_size must be an integer')
        if digest_size < 1:
            raise ValueError('digest_size must be a positive integer')
        self._digest_size = digest_size

    @property
    def digest_size(self) -> int:
        return self._digest_size