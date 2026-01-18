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
@event_class('Network.webSocketWillSendHandshakeRequest')
@dataclass
class WebSocketWillSendHandshakeRequest:
    """
    Fired when WebSocket is about to initiate handshake.
    """
    request_id: RequestId
    timestamp: MonotonicTime
    wall_time: TimeSinceEpoch
    request: WebSocketRequest

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketWillSendHandshakeRequest:
        return cls(request_id=RequestId.from_json(json['requestId']), timestamp=MonotonicTime.from_json(json['timestamp']), wall_time=TimeSinceEpoch.from_json(json['wallTime']), request=WebSocketRequest.from_json(json['request']))