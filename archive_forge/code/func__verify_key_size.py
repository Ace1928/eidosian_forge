from __future__ import annotations
from cryptography import utils
from cryptography.hazmat.primitives.ciphers import (
def _verify_key_size(algorithm: CipherAlgorithm, key: bytes) -> bytes:
    utils._check_byteslike('key', key)
    if len(key) * 8 not in algorithm.key_sizes:
        raise ValueError('Invalid key size ({}) for {}.'.format(len(key) * 8, algorithm.name))
    return key