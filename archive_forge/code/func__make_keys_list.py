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
def _make_keys_list(secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes]) -> list[bytes]:
    if isinstance(secret_key, (str, bytes)):
        return [want_bytes(secret_key)]
    return [want_bytes(s) for s in secret_key]