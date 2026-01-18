from __future__ import annotations
from cryptography import utils
from cryptography.hazmat.primitives.ciphers import (
class Camellia(BlockCipherAlgorithm):
    name = 'camellia'
    block_size = 128
    key_sizes = frozenset([128, 192, 256])

    def __init__(self, key: bytes):
        self.key = _verify_key_size(self, key)

    @property
    def key_size(self) -> int:
        return len(self.key) * 8