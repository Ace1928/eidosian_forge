from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
@dataclass
class CoverageRange:
    """
    Coverage data for a source range.
    """
    start_offset: int
    end_offset: int
    count: int

    def to_json(self):
        json = dict()
        json['startOffset'] = self.start_offset
        json['endOffset'] = self.end_offset
        json['count'] = self.count
        return json

    @classmethod
    def from_json(cls, json):
        return cls(start_offset=int(json['startOffset']), end_offset=int(json['endOffset']), count=int(json['count']))