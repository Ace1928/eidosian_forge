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
def _validate_aud(self, payload: dict[str, Any], audience: str | Iterable[str] | None, *, strict: bool=False) -> None:
    if audience is None:
        if 'aud' not in payload or not payload['aud']:
            return
        raise InvalidAudienceError('Invalid audience')
    if 'aud' not in payload or not payload['aud']:
        raise MissingRequiredClaimError('aud')
    audience_claims = payload['aud']
    if strict:
        if not isinstance(audience, str):
            raise InvalidAudienceError('Invalid audience (strict)')
        if not isinstance(audience_claims, str):
            raise InvalidAudienceError('Invalid claim format in token (strict)')
        if audience != audience_claims:
            raise InvalidAudienceError("Audience doesn't match (strict)")
        return
    if isinstance(audience_claims, str):
        audience_claims = [audience_claims]
    if not isinstance(audience_claims, list):
        raise InvalidAudienceError('Invalid claim format in token')
    if any((not isinstance(c, str) for c in audience_claims)):
        raise InvalidAudienceError('Invalid claim format in token')
    if isinstance(audience, str):
        audience = [audience]
    if all((aud not in audience_claims for aud in audience)):
        raise InvalidAudienceError("Audience doesn't match")