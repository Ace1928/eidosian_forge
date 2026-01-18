from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@dataclass
class AXProperty:
    name: AXPropertyName
    value: AXValue

    def to_json(self):
        json = dict()
        json['name'] = self.name.to_json()
        json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=AXPropertyName.from_json(json['name']), value=AXValue.from_json(json['value']))