from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class ShorthandEntry:
    name: str
    value: str
    important: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        if self.important is not None:
            json['important'] = self.important
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), value=str(json['value']), important=bool(json['important']) if 'important' in json else None)