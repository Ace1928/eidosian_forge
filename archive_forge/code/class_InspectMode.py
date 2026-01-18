from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
class InspectMode(enum.Enum):
    SEARCH_FOR_NODE = 'searchForNode'
    SEARCH_FOR_UA_SHADOW_DOM = 'searchForUAShadowDOM'
    CAPTURE_AREA_SCREENSHOT = 'captureAreaScreenshot'
    SHOW_DISTANCES = 'showDistances'
    NONE = 'none'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)