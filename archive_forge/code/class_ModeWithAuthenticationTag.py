from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
class ModeWithAuthenticationTag(Mode, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def tag(self) -> typing.Optional[bytes]:
        """
        The value of the tag supplied to the constructor of this mode.
        """