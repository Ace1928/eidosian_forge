from __future__ import annotations
import abc
import typing
from cryptography.exceptions import (
from cryptography.hazmat.primitives._cipheralgorithm import CipherAlgorithm
from cryptography.hazmat.primitives.ciphers import modes
class AEADDecryptionContext(AEADCipherContext, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def finalize_with_tag(self, tag: bytes) -> bytes:
        """
        Returns the results of processing the final block as bytes and allows
        delayed passing of the authentication tag.
        """