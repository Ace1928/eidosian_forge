from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class SourceOrderConfig:
    """
    Configuration data for drawing the source order of an elements children.
    """
    parent_outline_color: dom.RGBA
    child_outline_color: dom.RGBA

    def to_json(self):
        json = dict()
        json['parentOutlineColor'] = self.parent_outline_color.to_json()
        json['childOutlineColor'] = self.child_outline_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(parent_outline_color=dom.RGBA.from_json(json['parentOutlineColor']), child_outline_color=dom.RGBA.from_json(json['childOutlineColor']))