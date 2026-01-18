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
@event_class('Network.webSocketClosed')
@dataclass
class WebSocketClosed:
    """
    Fired when WebSocket is closed.
    """
    request_id: RequestId
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketClosed:
        return cls(request_id=RequestId.from_json(json['requestId']), timestamp=MonotonicTime.from_json(json['timestamp']))