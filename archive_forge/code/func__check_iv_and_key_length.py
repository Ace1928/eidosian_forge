from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
def _check_iv_and_key_length(self: ModeWithInitializationVector, algorithm: CipherAlgorithm) -> None:
    if not isinstance(algorithm, BlockCipherAlgorithm):
        raise UnsupportedAlgorithm(f'{self} requires a block cipher algorithm', _Reasons.UNSUPPORTED_CIPHER)
    _check_aes_key_length(self, algorithm)
    _check_iv_length(self, algorithm)