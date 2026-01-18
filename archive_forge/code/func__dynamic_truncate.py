from __future__ import annotations
import base64
import typing
from urllib.parse import quote, urlencode
from cryptography.hazmat.primitives import constant_time, hmac
from cryptography.hazmat.primitives.hashes import SHA1, SHA256, SHA512
from cryptography.hazmat.primitives.twofactor import InvalidToken
def _dynamic_truncate(self, counter: int) -> int:
    ctx = hmac.HMAC(self._key, self._algorithm)
    ctx.update(counter.to_bytes(length=8, byteorder='big'))
    hmac_value = ctx.finalize()
    offset = hmac_value[len(hmac_value) - 1] & 15
    p = hmac_value[offset:offset + 4]
    return int.from_bytes(p, byteorder='big') & 2147483647