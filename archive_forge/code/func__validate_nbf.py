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
def _validate_nbf(self, payload: dict[str, Any], now: float, leeway: float) -> None:
    try:
        nbf = int(payload['nbf'])
    except ValueError:
        raise DecodeError('Not Before claim (nbf) must be an integer.')
    if nbf > now + leeway:
        raise ImmatureSignatureError('The token is not yet valid (nbf)')