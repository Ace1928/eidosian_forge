from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class ScreenOrientation:
    """
    Screen orientation.
    """
    type_: str
    angle: int

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['angle'] = self.angle
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=str(json['type']), angle=int(json['angle']))