from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
class WindowState(enum.Enum):
    """
    The state of the browser window.
    """
    NORMAL = 'normal'
    MINIMIZED = 'minimized'
    MAXIMIZED = 'maximized'
    FULLSCREEN = 'fullscreen'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)