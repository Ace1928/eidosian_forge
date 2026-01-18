from __future__ import annotations
import abc
import typing
from cryptography.exceptions import (
from cryptography.hazmat.primitives._cipheralgorithm import CipherAlgorithm
from cryptography.hazmat.primitives.ciphers import modes
class CipherContext(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def update(self, data: bytes) -> bytes:
        """
        Processes the provided bytes through the cipher and returns the results
        as bytes.
        """

    @abc.abstractmethod
    def update_into(self, data: bytes, buf: bytes) -> int:
        """
        Processes the provided bytes and writes the resulting data into the
        provided buffer. Returns the number of bytes written.
        """

    @abc.abstractmethod
    def finalize(self) -> bytes:
        """
        Returns the results of processing the final block as bytes.
        """