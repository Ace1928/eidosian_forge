from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat._oid import ObjectIdentifier
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class EllipticCurvePrivateNumbers:

    def __init__(self, private_value: int, public_numbers: EllipticCurvePublicNumbers):
        if not isinstance(private_value, int):
            raise TypeError('private_value must be an integer.')
        if not isinstance(public_numbers, EllipticCurvePublicNumbers):
            raise TypeError('public_numbers must be an EllipticCurvePublicNumbers instance.')
        self._private_value = private_value
        self._public_numbers = public_numbers

    def private_key(self, backend: typing.Any=None) -> EllipticCurvePrivateKey:
        from cryptography.hazmat.backends.openssl.backend import backend as ossl
        return ossl.load_elliptic_curve_private_numbers(self)

    @property
    def private_value(self) -> int:
        return self._private_value

    @property
    def public_numbers(self) -> EllipticCurvePublicNumbers:
        return self._public_numbers

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EllipticCurvePrivateNumbers):
            return NotImplemented
        return self.private_value == other.private_value and self.public_numbers == other.public_numbers

    def __hash__(self) -> int:
        return hash((self.private_value, self.public_numbers))