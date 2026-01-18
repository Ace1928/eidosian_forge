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
class TagMarkup(JSONTag):
    """Serialize anything matching the :class:`~markupsafe.Markup` API by
    having a ``__html__`` method to the result of that method. Always
    deserializes to an instance of :class:`~markupsafe.Markup`."""
    __slots__ = ()
    key = ' m'

    def check(self, value: t.Any) -> bool:
        return callable(getattr(value, '__html__', None))

    def to_json(self, value: t.Any) -> t.Any:
        return str(value.__html__())

    def to_python(self, value: t.Any) -> t.Any:
        return Markup(value)