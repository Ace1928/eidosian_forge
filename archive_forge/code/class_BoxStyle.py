from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class BoxStyle:
    """
    Style information for drawing a box.
    """
    fill_color: typing.Optional[dom.RGBA] = None
    hatch_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.fill_color is not None:
            json['fillColor'] = self.fill_color.to_json()
        if self.hatch_color is not None:
            json['hatchColor'] = self.hatch_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(fill_color=dom.RGBA.from_json(json['fillColor']) if 'fillColor' in json else None, hatch_color=dom.RGBA.from_json(json['hatchColor']) if 'hatchColor' in json else None)