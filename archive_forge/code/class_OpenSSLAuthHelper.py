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
class OpenSSLAuthHelper(AuthHelper, metaclass=abc.ABCMeta):
    """OpenSSL based public key based authentication helper for Ansible Core CI."""

    def sign_bytes(self, payload_bytes: bytes) -> bytes:
        """Sign the given payload and return the signature, initializing a new key pair if required."""
        private_key_pem = self.initialize_private_key()
        with tempfile.NamedTemporaryFile() as private_key_file:
            private_key_file.write(to_bytes(private_key_pem))
            private_key_file.flush()
            with tempfile.NamedTemporaryFile() as payload_file:
                payload_file.write(payload_bytes)
                payload_file.flush()
                with tempfile.NamedTemporaryFile() as signature_file:
                    raw_command(['openssl', 'dgst', '-sha256', '-sign', private_key_file.name, '-out', signature_file.name, payload_file.name], capture=True)
                    signature_raw_bytes = signature_file.read()
        return signature_raw_bytes

    def generate_private_key(self) -> str:
        """Generate a new key pair, publishing the public key and returning the private key."""
        private_key_pem = raw_command(['openssl', 'ecparam', '-genkey', '-name', 'secp384r1', '-noout'], capture=True)[0]
        public_key_pem = raw_command(['openssl', 'ec', '-pubout'], data=private_key_pem, capture=True)[0]
        self.publish_public_key(public_key_pem)
        return private_key_pem