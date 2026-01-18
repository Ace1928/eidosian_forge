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
@event_class('Page.frameClearedScheduledNavigation')
@dataclass
class FrameClearedScheduledNavigation:
    """
    Fired when frame no longer has a scheduled navigation.
    """
    frame_id: FrameId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameClearedScheduledNavigation:
        return cls(frame_id=FrameId.from_json(json['frameId']))