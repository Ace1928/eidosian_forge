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
@dataclass
class FrameTree:
    """
    Information about the Frame hierarchy.
    """
    frame: Frame
    child_frames: typing.Optional[typing.List[FrameTree]] = None

    def to_json(self):
        json = dict()
        json['frame'] = self.frame.to_json()
        if self.child_frames is not None:
            json['childFrames'] = [i.to_json() for i in self.child_frames]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(frame=Frame.from_json(json['frame']), child_frames=[FrameTree.from_json(i) for i in json['childFrames']] if 'childFrames' in json else None)