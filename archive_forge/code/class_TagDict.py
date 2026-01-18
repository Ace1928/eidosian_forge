from __future__ import annotations
import typing as t
from base64 import b64decode
from base64 import b64encode
from datetime import datetime
from uuid import UUID
from markupsafe import Markup
from werkzeug.http import http_date
from werkzeug.http import parse_date
from ..json import dumps
from ..json import loads
class TagDict(JSONTag):
    """Tag for 1-item dicts whose only key matches a registered tag.

    Internally, the dict key is suffixed with `__`, and the suffix is removed
    when deserializing.
    """
    __slots__ = ()
    key = ' di'

    def check(self, value: t.Any) -> bool:
        return isinstance(value, dict) and len(value) == 1 and (next(iter(value)) in self.serializer.tags)

    def to_json(self, value: t.Any) -> t.Any:
        key = next(iter(value))
        return {f'{key}__': self.serializer.tag(value[key])}

    def to_python(self, value: t.Any) -> t.Any:
        key = next(iter(value))
        return {key[:-2]: value[key]}