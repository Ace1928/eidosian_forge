from __future__ import annotations
from cryptography import utils
from cryptography.hazmat.primitives.ciphers import (
class AES128(BlockCipherAlgorithm):
    name = 'AES'
    block_size = 128
    key_sizes = frozenset([128])
    key_size = 128

    def __init__(self, key: bytes):
        self.key = _verify_key_size(self, key)