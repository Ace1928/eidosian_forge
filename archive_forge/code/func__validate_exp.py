from __future__ import annotations
import json
import warnings
from calendar import timegm
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from . import api_jws
from .exceptions import (
from .warnings import RemovedInPyjwt3Warning
def _validate_exp(self, payload: dict[str, Any], now: float, leeway: float) -> None:
    try:
        exp = int(payload['exp'])
    except ValueError:
        raise DecodeError('Expiration Time claim (exp) must be an integer.')
    if exp <= now - leeway:
        raise ExpiredSignatureError('Signature has expired')