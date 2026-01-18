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
class WebSocketRequest:
    """
    WebSocket request data.
    """
    headers: Headers

    def to_json(self):
        json = dict()
        json['headers'] = self.headers.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(headers=Headers.from_json(json['headers']))