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
@event_class('Page.frameNavigated')
@dataclass
class FrameNavigated:
    """
    Fired once navigation of the frame has completed. Frame is now associated with the new loader.
    """
    frame: Frame
    type_: NavigationType

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameNavigated:
        return cls(frame=Frame.from_json(json['frame']), type_=NavigationType.from_json(json['type']))