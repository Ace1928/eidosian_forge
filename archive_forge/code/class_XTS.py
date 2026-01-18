from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
class XTS(ModeWithTweak):
    name = 'XTS'

    def __init__(self, tweak: bytes):
        utils._check_byteslike('tweak', tweak)
        if len(tweak) != 16:
            raise ValueError('tweak must be 128-bits (16 bytes)')
        self._tweak = tweak

    @property
    def tweak(self) -> bytes:
        return self._tweak

    def validate_for_algorithm(self, algorithm: CipherAlgorithm) -> None:
        if isinstance(algorithm, (algorithms.AES128, algorithms.AES256)):
            raise TypeError('The AES128 and AES256 classes do not support XTS, please use the standard AES class instead.')
        if algorithm.key_size not in (256, 512):
            raise ValueError('The XTS specification requires a 256-bit key for AES-128-XTS and 512-bit key for AES-256-XTS')