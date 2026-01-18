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
@event_class('Network.resourceChangedPriority')
@dataclass
class ResourceChangedPriority:
    """
    **EXPERIMENTAL**

    Fired when resource loading priority is changed
    """
    request_id: RequestId
    new_priority: ResourcePriority
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResourceChangedPriority:
        return cls(request_id=RequestId.from_json(json['requestId']), new_priority=ResourcePriority.from_json(json['newPriority']), timestamp=MonotonicTime.from_json(json['timestamp']))