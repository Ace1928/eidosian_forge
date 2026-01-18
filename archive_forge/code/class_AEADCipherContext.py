from __future__ import annotations
import abc
import typing
from cryptography.exceptions import (
from cryptography.hazmat.primitives._cipheralgorithm import CipherAlgorithm
from cryptography.hazmat.primitives.ciphers import modes
class AEADCipherContext(CipherContext, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def authenticate_additional_data(self, data: bytes) -> None:
        """
        Authenticates the provided bytes.
        """