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
@event_class('Network.webSocketFrameError')
@dataclass
class WebSocketFrameError:
    """
    Fired when WebSocket message error occurs.
    """
    request_id: RequestId
    timestamp: MonotonicTime
    error_message: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketFrameError:
        return cls(request_id=RequestId.from_json(json['requestId']), timestamp=MonotonicTime.from_json(json['timestamp']), error_message=str(json['errorMessage']))