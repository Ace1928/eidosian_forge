from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
def _check_iv_length(self: ModeWithInitializationVector, algorithm: BlockCipherAlgorithm) -> None:
    if len(self.initialization_vector) * 8 != algorithm.block_size:
        raise ValueError('Invalid IV size ({}) for {}.'.format(len(self.initialization_vector), self.name))