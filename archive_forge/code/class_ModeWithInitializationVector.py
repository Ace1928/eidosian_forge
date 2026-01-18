from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
class ModeWithInitializationVector(Mode, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def initialization_vector(self) -> bytes:
        """
        The value of the initialization vector for this mode as bytes.
        """