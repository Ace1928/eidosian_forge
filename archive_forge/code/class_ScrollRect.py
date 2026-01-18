from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
@dataclass
class ScrollRect:
    """
    Rectangle where scrolling happens on the main thread.
    """
    rect: dom.Rect
    type_: str

    def to_json(self):
        json = dict()
        json['rect'] = self.rect.to_json()
        json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(rect=dom.Rect.from_json(json['rect']), type_=str(json['type']))