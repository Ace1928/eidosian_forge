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
class BlockedSetCookieWithReason:
    """
    A cookie which was not stored from a response with the corresponding reason.
    """
    blocked_reasons: typing.List[SetCookieBlockedReason]
    cookie_line: str
    cookie: typing.Optional[Cookie] = None

    def to_json(self):
        json = dict()
        json['blockedReasons'] = [i.to_json() for i in self.blocked_reasons]
        json['cookieLine'] = self.cookie_line
        if self.cookie is not None:
            json['cookie'] = self.cookie.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(blocked_reasons=[SetCookieBlockedReason.from_json(i) for i in json['blockedReasons']], cookie_line=str(json['cookieLine']), cookie=Cookie.from_json(json['cookie']) if 'cookie' in json else None)