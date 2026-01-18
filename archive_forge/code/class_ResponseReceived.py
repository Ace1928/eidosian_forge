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
@event_class('Network.responseReceived')
@dataclass
class ResponseReceived:
    """
    Fired when HTTP response is available.
    """
    request_id: RequestId
    loader_id: LoaderId
    timestamp: MonotonicTime
    type_: ResourceType
    response: Response
    has_extra_info: bool
    frame_id: typing.Optional[page.FrameId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResponseReceived:
        return cls(request_id=RequestId.from_json(json['requestId']), loader_id=LoaderId.from_json(json['loaderId']), timestamp=MonotonicTime.from_json(json['timestamp']), type_=ResourceType.from_json(json['type']), response=Response.from_json(json['response']), has_extra_info=bool(json['hasExtraInfo']), frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None)