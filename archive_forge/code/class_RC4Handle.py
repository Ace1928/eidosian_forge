import base64
import binascii
import hashlib
import hmac
import io
import re
import struct
import typing
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from spnego._ntlm_raw.des import DES
from spnego._ntlm_raw.md4 import md4
from spnego._ntlm_raw.messages import (
class RC4Handle:
    """RC4 class to wrap the underlying crypto function."""

    def __init__(self, key: bytes) -> None:
        self._key = key
        arc4 = algorithms.ARC4(self._key)
        cipher = Cipher(arc4, mode=None, backend=default_backend())
        self._handle = cipher.encryptor()

    def update(self, b_data: bytes) -> bytes:
        """Update the RC4 stream and return the encrypted/decrypted bytes."""
        return self._handle.update(b_data)

    def reset(self) -> None:
        """Reset's the cipher stream back to the original state."""
        arc4 = algorithms.ARC4(self._key)
        cipher = Cipher(arc4, mode=None, backend=default_backend())
        self._handle = cipher.encryptor()