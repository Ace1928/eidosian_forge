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
def _ensure_bytes(secret_key: str | bytes | None) -> bytes | None:
    if secret_key is None:
        return None
    elif isinstance(secret_key, bytes):
        return secret_key
    else:
        return codecs.encode(secret_key, 'utf-8')