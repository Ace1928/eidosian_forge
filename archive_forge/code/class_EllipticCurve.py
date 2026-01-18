from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat._oid import ObjectIdentifier
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class EllipticCurve(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        The name of the curve. e.g. secp256r1.
        """

    @property
    @abc.abstractmethod
    def key_size(self) -> int:
        """
        Bit size of a secret scalar for the curve.
        """