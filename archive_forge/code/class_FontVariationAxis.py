from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class FontVariationAxis:
    """
    Information about font variation axes for variable fonts
    """
    tag: str
    name: str
    min_value: float
    max_value: float
    default_value: float

    def to_json(self):
        json = dict()
        json['tag'] = self.tag
        json['name'] = self.name
        json['minValue'] = self.min_value
        json['maxValue'] = self.max_value
        json['defaultValue'] = self.default_value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(tag=str(json['tag']), name=str(json['name']), min_value=float(json['minValue']), max_value=float(json['maxValue']), default_value=float(json['defaultValue']))