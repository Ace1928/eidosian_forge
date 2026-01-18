from __future__ import annotations
import abc
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization
class X448PublicKey(metaclass=abc.ABCMeta):

    @classmethod
    def from_public_bytes(cls, data: bytes) -> X448PublicKey:
        from cryptography.hazmat.backends.openssl.backend import backend
        if not backend.x448_supported():
            raise UnsupportedAlgorithm('X448 is not supported by this version of OpenSSL.', _Reasons.UNSUPPORTED_EXCHANGE_ALGORITHM)
        return backend.x448_load_public_bytes(data)

    @abc.abstractmethod
    def public_bytes(self, encoding: _serialization.Encoding, format: _serialization.PublicFormat) -> bytes:
        """
        The serialized bytes of the public key.
        """

    @abc.abstractmethod
    def public_bytes_raw(self) -> bytes:
        """
        The raw bytes of the public key.
        Equivalent to public_bytes(Raw, Raw).
        """

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Checks equality.
        """