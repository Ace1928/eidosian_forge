from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@event_class('Page.frameDetached')
@dataclass
class FrameDetached:
    """
    Fired when frame has been detached from its parent.
    """
    frame_id: FrameId
    reason: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameDetached:
        return cls(frame_id=FrameId.from_json(json['frameId']), reason=str(json['reason']))