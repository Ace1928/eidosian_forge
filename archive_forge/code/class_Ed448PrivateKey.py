from __future__ import annotations
import abc
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization
class Ed448PrivateKey(metaclass=abc.ABCMeta):

    @classmethod
    def generate(cls) -> Ed448PrivateKey:
        from cryptography.hazmat.backends.openssl.backend import backend
        if not backend.ed448_supported():
            raise UnsupportedAlgorithm('ed448 is not supported by this version of OpenSSL.', _Reasons.UNSUPPORTED_PUBLIC_KEY_ALGORITHM)
        return backend.ed448_generate_key()

    @classmethod
    def from_private_bytes(cls, data: bytes) -> Ed448PrivateKey:
        from cryptography.hazmat.backends.openssl.backend import backend
        if not backend.ed448_supported():
            raise UnsupportedAlgorithm('ed448 is not supported by this version of OpenSSL.', _Reasons.UNSUPPORTED_PUBLIC_KEY_ALGORITHM)
        return backend.ed448_load_private_bytes(data)

    @abc.abstractmethod
    def public_key(self) -> Ed448PublicKey:
        """
        The Ed448PublicKey derived from the private key.
        """

    @abc.abstractmethod
    def sign(self, data: bytes) -> bytes:
        """
        Signs the data.
        """

    @abc.abstractmethod
    def private_bytes(self, encoding: _serialization.Encoding, format: _serialization.PrivateFormat, encryption_algorithm: _serialization.KeySerializationEncryption) -> bytes:
        """
        The serialized bytes of the private key.
        """

    @abc.abstractmethod
    def private_bytes_raw(self) -> bytes:
        """
        The raw bytes of the private key.
        Equivalent to private_bytes(Raw, Raw, NoEncryption()).
        """