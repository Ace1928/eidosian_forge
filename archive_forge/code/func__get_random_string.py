from __future__ import annotations
import logging # isort:skip
import base64
import calendar
import codecs
import datetime as dt
import hashlib
import hmac
import json
import time
import zlib
from typing import TYPE_CHECKING, Any
from ..core.types import ID
from ..settings import settings
from .warnings import warn
def _get_random_string(length: int=44, allowed_chars: str='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', secret_key: bytes | None=settings.secret_key_bytes()) -> str:
    """ Return a securely generated random string.

    With the a-z, A-Z, 0-9 character set:
    Length 12 is a 71-bit value. log_2((26+26+10)^12) =~ 71
    Length 44 is a 261-bit value. log_2((26+26+10)^44) = 261

    """
    secret_key = _ensure_bytes(secret_key)
    _reseed_if_needed(using_sysrandom, secret_key)
    return ''.join((random.choice(allowed_chars) for _ in range(length)))