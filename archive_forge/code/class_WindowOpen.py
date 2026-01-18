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
@event_class('Page.windowOpen')
@dataclass
class WindowOpen:
    """
    Fired when a new window is going to be opened, via window.open(), link click, form submission,
    etc.
    """
    url: str
    window_name: str
    window_features: typing.List[str]
    user_gesture: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WindowOpen:
        return cls(url=str(json['url']), window_name=str(json['windowName']), window_features=[str(i) for i in json['windowFeatures']], user_gesture=bool(json['userGesture']))