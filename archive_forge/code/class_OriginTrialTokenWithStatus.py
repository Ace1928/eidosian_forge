from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@dataclass
class OriginTrialTokenWithStatus:
    raw_token_text: str
    status: OriginTrialTokenStatus
    parsed_token: typing.Optional[OriginTrialToken] = None

    def to_json(self):
        json = dict()
        json['rawTokenText'] = self.raw_token_text
        json['status'] = self.status.to_json()
        if self.parsed_token is not None:
            json['parsedToken'] = self.parsed_token.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(raw_token_text=str(json['rawTokenText']), status=OriginTrialTokenStatus.from_json(json['status']), parsed_token=OriginTrialToken.from_json(json['parsedToken']) if 'parsedToken' in json else None)