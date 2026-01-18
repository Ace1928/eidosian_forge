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
@event_class('Network.eventSourceMessageReceived')
@dataclass
class EventSourceMessageReceived:
    """
    Fired when EventSource message is received.
    """
    request_id: RequestId
    timestamp: MonotonicTime
    event_name: str
    event_id: str
    data: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> EventSourceMessageReceived:
        return cls(request_id=RequestId.from_json(json['requestId']), timestamp=MonotonicTime.from_json(json['timestamp']), event_name=str(json['eventName']), event_id=str(json['eventId']), data=str(json['data']))