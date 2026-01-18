from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
def _check_nonce_length(nonce: bytes, name: str, algorithm: CipherAlgorithm) -> None:
    if not isinstance(algorithm, BlockCipherAlgorithm):
        raise UnsupportedAlgorithm(f'{name} requires a block cipher algorithm', _Reasons.UNSUPPORTED_CIPHER)
    if len(nonce) * 8 != algorithm.block_size:
        raise ValueError(f'Invalid nonce size ({len(nonce)}) for {name}.')