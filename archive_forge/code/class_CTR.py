from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
class CTR(ModeWithNonce):
    name = 'CTR'

    def __init__(self, nonce: bytes):
        utils._check_byteslike('nonce', nonce)
        self._nonce = nonce

    @property
    def nonce(self) -> bytes:
        return self._nonce

    def validate_for_algorithm(self, algorithm: CipherAlgorithm) -> None:
        _check_aes_key_length(self, algorithm)
        _check_nonce_length(self.nonce, self.name, algorithm)