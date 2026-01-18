from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@dataclass
class BoxModel:
    """
    Box model.
    """
    content: Quad
    padding: Quad
    border: Quad
    margin: Quad
    width: int
    height: int
    shape_outside: typing.Optional[ShapeOutsideInfo] = None

    def to_json(self):
        json = dict()
        json['content'] = self.content.to_json()
        json['padding'] = self.padding.to_json()
        json['border'] = self.border.to_json()
        json['margin'] = self.margin.to_json()
        json['width'] = self.width
        json['height'] = self.height
        if self.shape_outside is not None:
            json['shapeOutside'] = self.shape_outside.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(content=Quad.from_json(json['content']), padding=Quad.from_json(json['padding']), border=Quad.from_json(json['border']), margin=Quad.from_json(json['margin']), width=int(json['width']), height=int(json['height']), shape_outside=ShapeOutsideInfo.from_json(json['shapeOutside']) if 'shapeOutside' in json else None)