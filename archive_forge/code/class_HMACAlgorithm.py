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
class HMACAlgorithm(SigningAlgorithm):
    """Provides signature generation using HMACs."""
    default_digest_method: t.Any = staticmethod(_lazy_sha1)

    def __init__(self, digest_method: t.Any=None):
        if digest_method is None:
            digest_method = self.default_digest_method
        self.digest_method: t.Any = digest_method

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        mac = hmac.new(key, msg=value, digestmod=self.digest_method)
        return mac.digest()