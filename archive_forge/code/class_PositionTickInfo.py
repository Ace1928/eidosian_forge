from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
@dataclass
class PositionTickInfo:
    """
    Specifies a number of samples attributed to a certain source position.
    """
    line: int
    ticks: int

    def to_json(self):
        json = dict()
        json['line'] = self.line
        json['ticks'] = self.ticks
        return json

    @classmethod
    def from_json(cls, json):
        return cls(line=int(json['line']), ticks=int(json['ticks']))