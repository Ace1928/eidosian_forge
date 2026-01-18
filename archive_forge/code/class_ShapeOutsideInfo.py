from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@dataclass
class ShapeOutsideInfo:
    """
    CSS Shape Outside details.
    """
    bounds: Quad
    shape: typing.List[typing.Any]
    margin_shape: typing.List[typing.Any]

    def to_json(self):
        json = dict()
        json['bounds'] = self.bounds.to_json()
        json['shape'] = [i for i in self.shape]
        json['marginShape'] = [i for i in self.margin_shape]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(bounds=Quad.from_json(json['bounds']), shape=[i for i in json['shape']], margin_shape=[i for i in json['marginShape']])