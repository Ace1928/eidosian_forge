from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@dataclass
class SearchMatch:
    """
    Search match for resource.
    """
    line_number: float
    line_content: str

    def to_json(self):
        json = dict()
        json['lineNumber'] = self.line_number
        json['lineContent'] = self.line_content
        return json

    @classmethod
    def from_json(cls, json):
        return cls(line_number=float(json['lineNumber']), line_content=str(json['lineContent']))