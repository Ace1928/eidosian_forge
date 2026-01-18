from __future__ import annotations
import collections.abc as cabc
import hashlib
import hmac
import typing as t
from .encoding import _base64_alphabet
from .encoding import base64_decode
from .encoding import base64_encode
from .encoding import want_bytes
from .exc import BadSignature
class SigningAlgorithm:
    """Subclasses must implement :meth:`get_signature` to provide
    signature generation functionality.
    """

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        """Returns the signature for the given key and value."""
        raise NotImplementedError()

    def verify_signature(self, key: bytes, value: bytes, sig: bytes) -> bool:
        """Verifies the given signature matches the expected
        signature.
        """
        return hmac.compare_digest(sig, self.get_signature(key, value))