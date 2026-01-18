from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat._oid import ObjectIdentifier
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class EllipticCurvePrivateKey(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def exchange(self, algorithm: ECDH, peer_public_key: EllipticCurvePublicKey) -> bytes:
        """
        Performs a key exchange operation using the provided algorithm with the
        provided peer's public key.
        """

    @abc.abstractmethod
    def public_key(self) -> EllipticCurvePublicKey:
        """
        The EllipticCurvePublicKey for this private key.
        """

    @property
    @abc.abstractmethod
    def curve(self) -> EllipticCurve:
        """
        The EllipticCurve that this key is on.
        """

    @property
    @abc.abstractmethod
    def key_size(self) -> int:
        """
        Bit size of a secret scalar for the curve.
        """

    @abc.abstractmethod
    def sign(self, data: bytes, signature_algorithm: EllipticCurveSignatureAlgorithm) -> bytes:
        """
        Signs the data
        """

    @abc.abstractmethod
    def private_numbers(self) -> EllipticCurvePrivateNumbers:
        """
        Returns an EllipticCurvePrivateNumbers.
        """

    @abc.abstractmethod
    def private_bytes(self, encoding: _serialization.Encoding, format: _serialization.PrivateFormat, encryption_algorithm: _serialization.KeySerializationEncryption) -> bytes:
        """
        Returns the key serialized as bytes.
        """