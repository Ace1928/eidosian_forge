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
class WebSocketFrame:
    """
    WebSocket message data. This represents an entire WebSocket message, not just a fragmented frame as the name suggests.
    """
    opcode: float
    mask: bool
    payload_data: str

    def to_json(self):
        json = dict()
        json['opcode'] = self.opcode
        json['mask'] = self.mask
        json['payloadData'] = self.payload_data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(opcode=float(json['opcode']), mask=bool(json['mask']), payload_data=str(json['payloadData']))