from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class GestureSourceType(enum.Enum):
    DEFAULT = 'default'
    TOUCH = 'touch'
    MOUSE = 'mouse'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)