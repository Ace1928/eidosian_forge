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
class BackForwardCacheNotRestoredExplanation:
    type_: BackForwardCacheNotRestoredReasonType
    reason: BackForwardCacheNotRestoredReason
    context: typing.Optional[str] = None
    details: typing.Optional[typing.List[BackForwardCacheBlockingDetails]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        json['reason'] = self.reason.to_json()
        if self.context is not None:
            json['context'] = self.context
        if self.details is not None:
            json['details'] = [i.to_json() for i in self.details]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=BackForwardCacheNotRestoredReasonType.from_json(json['type']), reason=BackForwardCacheNotRestoredReason.from_json(json['reason']), context=str(json['context']) if 'context' in json else None, details=[BackForwardCacheBlockingDetails.from_json(i) for i in json['details']] if 'details' in json else None)