from __future__ import annotations
import hashlib
import typing as t
from collections.abc import MutableMapping
from datetime import datetime
from datetime import timezone
from itsdangerous import BadSignature
from itsdangerous import URLSafeTimedSerializer
from werkzeug.datastructures import CallbackDict
from .json.tag import TaggedJSONSerializer
class SessionMixin(MutableMapping):
    """Expands a basic dictionary with session attributes."""

    @property
    def permanent(self) -> bool:
        """This reflects the ``'_permanent'`` key in the dict."""
        return self.get('_permanent', False)

    @permanent.setter
    def permanent(self, value: bool) -> None:
        self['_permanent'] = bool(value)
    new = False
    modified = True
    accessed = True