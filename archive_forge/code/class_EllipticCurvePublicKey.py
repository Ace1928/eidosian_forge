from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat._oid import ObjectIdentifier
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class EllipticCurvePublicKey(metaclass=abc.ABCMeta):

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
    def public_numbers(self) -> EllipticCurvePublicNumbers:
        """
        Returns an EllipticCurvePublicNumbers.
        """

    @abc.abstractmethod
    def public_bytes(self, encoding: _serialization.Encoding, format: _serialization.PublicFormat) -> bytes:
        """
        Returns the key serialized as bytes.
        """

    @abc.abstractmethod
    def verify(self, signature: bytes, data: bytes, signature_algorithm: EllipticCurveSignatureAlgorithm) -> None:
        """
        Verifies the signature of the data.
        """

    @classmethod
    def from_encoded_point(cls, curve: EllipticCurve, data: bytes) -> EllipticCurvePublicKey:
        utils._check_bytes('data', data)
        if not isinstance(curve, EllipticCurve):
            raise TypeError('curve must be an EllipticCurve instance')
        if len(data) == 0:
            raise ValueError('data must not be an empty byte string')
        if data[0] not in [2, 3, 4]:
            raise ValueError('Unsupported elliptic curve point type')
        from cryptography.hazmat.backends.openssl.backend import backend
        return backend.load_elliptic_curve_public_bytes(curve, data)

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Checks equality.
        """