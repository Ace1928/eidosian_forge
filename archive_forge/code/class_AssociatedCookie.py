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
class AssociatedCookie:
    """
    A cookie associated with the request which may or may not be sent with it.
    Includes the cookies itself and reasons for blocking or exemption.
    """
    cookie: Cookie
    blocked_reasons: typing.List[CookieBlockedReason]
    exemption_reason: typing.Optional[CookieExemptionReason] = None

    def to_json(self):
        json = dict()
        json['cookie'] = self.cookie.to_json()
        json['blockedReasons'] = [i.to_json() for i in self.blocked_reasons]
        if self.exemption_reason is not None:
            json['exemptionReason'] = self.exemption_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(cookie=Cookie.from_json(json['cookie']), blocked_reasons=[CookieBlockedReason.from_json(i) for i in json['blockedReasons']], exemption_reason=CookieExemptionReason.from_json(json['exemptionReason']) if 'exemptionReason' in json else None)