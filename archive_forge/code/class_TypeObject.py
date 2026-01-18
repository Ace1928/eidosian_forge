from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
@dataclass
class TypeObject:
    """
    Describes a type collected during runtime.
    """
    name: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']))