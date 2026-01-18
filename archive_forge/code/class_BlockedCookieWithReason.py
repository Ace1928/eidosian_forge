from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
@dataclass
class BlockedCookieWithReason:
    """
    A cookie with was not sent with a request with the corresponding reason.
    """
    blocked_reasons: typing.List[CookieBlockedReason]
    cookie: Cookie

    def to_json(self):
        json = dict()
        json['blockedReasons'] = [i.to_json() for i in self.blocked_reasons]
        json['cookie'] = self.cookie.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(blocked_reasons=[CookieBlockedReason.from_json(i) for i in json['blockedReasons']], cookie=Cookie.from_json(json['cookie']))