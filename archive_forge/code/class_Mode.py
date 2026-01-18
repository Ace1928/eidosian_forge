from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
class Mode(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        A string naming this mode (e.g. "ECB", "CBC").
        """

    @abc.abstractmethod
    def validate_for_algorithm(self, algorithm: CipherAlgorithm) -> None:
        """
        Checks that all the necessary invariants of this (mode, algorithm)
        combination are met.
        """