from __future__ import annotations
import abc
import base64
import json
import os
import tempfile
import typing as t
from ..encoding import (
from ..io import (
from ..config import (
from ..util import (
class CryptographyAuthHelper(AuthHelper, metaclass=abc.ABCMeta):
    """Cryptography based public key based authentication helper for Ansible Core CI."""

    def sign_bytes(self, payload_bytes: bytes) -> bytes:
        """Sign the given payload and return the signature, initializing a new key pair if required."""
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        private_key_pem = self.initialize_private_key()
        private_key = load_pem_private_key(to_bytes(private_key_pem), None, default_backend())
        assert isinstance(private_key, ec.EllipticCurvePrivateKey)
        signature_raw_bytes = private_key.sign(payload_bytes, ec.ECDSA(hashes.SHA256()))
        return signature_raw_bytes

    def generate_private_key(self) -> str:
        """Generate a new key pair, publishing the public key and returning the private key."""
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        private_key = ec.generate_private_key(ec.SECP384R1(), default_backend())
        public_key = private_key.public_key()
        private_key_pem = to_text(private_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption()))
        public_key_pem = to_text(public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo))
        self.publish_public_key(public_key_pem)
        return private_key_pem