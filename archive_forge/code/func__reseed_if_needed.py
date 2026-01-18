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
def _reseed_if_needed(using_sysrandom: bool, secret_key: bytes | None) -> None:
    secret_key = _ensure_bytes(secret_key)
    if not using_sysrandom:
        data = f'{random.getstate()}{time.time()}{secret_key!s}'.encode()
        random.seed(hashlib.sha256(data).digest())