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
def _decode_payload(self, decoded: dict[str, Any]) -> Any:
    """
        Decode the payload from a JWS dictionary (payload, signature, header).

        This method is intended to be overridden by subclasses that need to
        decode the payload in a different way, e.g. decompress compressed
        payloads.
        """
    try:
        payload = json.loads(decoded['payload'])
    except ValueError as e:
        raise DecodeError(f'Invalid payload string: {e}')
    if not isinstance(payload, dict):
        raise DecodeError('Invalid payload string: must be a json object')
    return payload