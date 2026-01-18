from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@dataclass
class ScriptPosition:
    """
    Location in the source code.
    """
    line_number: int
    column_number: int

    def to_json(self):
        json = dict()
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(line_number=int(json['lineNumber']), column_number=int(json['columnNumber']))