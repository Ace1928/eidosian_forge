from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@dataclass
class KeyframeStyle:
    """
    Keyframe Style
    """
    offset: str
    easing: str

    def to_json(self):
        json = dict()
        json['offset'] = self.offset
        json['easing'] = self.easing
        return json

    @classmethod
    def from_json(cls, json):
        return cls(offset=str(json['offset']), easing=str(json['easing']))