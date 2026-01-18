from __future__ import annotations
from vine import transform
from .message import AsyncMessage
def _coerce_field_value(self, key, type, response):
    return type(response[key])