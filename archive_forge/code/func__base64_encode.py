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
def _base64_encode(decoded: bytes | str) -> str:
    decoded_as_bytes = _ensure_bytes(decoded)
    encoded = codecs.decode(base64.urlsafe_b64encode(decoded_as_bytes), 'ascii')
    return str(encoded.rstrip('='))