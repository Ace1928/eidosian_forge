from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class Ctap2Version(enum.Enum):
    CTAP2_0 = 'ctap2_0'
    CTAP2_1 = 'ctap2_1'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)