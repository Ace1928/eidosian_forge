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
@event_class('Network.requestServedFromCache')
@dataclass
class RequestServedFromCache:
    """
    Fired if request ended up loading from cache.
    """
    request_id: RequestId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestServedFromCache:
        return cls(request_id=RequestId.from_json(json['requestId']))