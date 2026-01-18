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
@event_class('Network.loadingFailed')
@dataclass
class LoadingFailed:
    """
    Fired when HTTP request has failed to load.
    """
    request_id: RequestId
    timestamp: MonotonicTime
    type_: ResourceType
    error_text: str
    canceled: typing.Optional[bool]
    blocked_reason: typing.Optional[BlockedReason]
    cors_error_status: typing.Optional[CorsErrorStatus]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadingFailed:
        return cls(request_id=RequestId.from_json(json['requestId']), timestamp=MonotonicTime.from_json(json['timestamp']), type_=ResourceType.from_json(json['type']), error_text=str(json['errorText']), canceled=bool(json['canceled']) if 'canceled' in json else None, blocked_reason=BlockedReason.from_json(json['blockedReason']) if 'blockedReason' in json else None, cors_error_status=CorsErrorStatus.from_json(json['corsErrorStatus']) if 'corsErrorStatus' in json else None)