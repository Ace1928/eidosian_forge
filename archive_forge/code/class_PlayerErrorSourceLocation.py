from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class PlayerErrorSourceLocation:
    """
    Represents logged source line numbers reported in an error.
    NOTE: file and line are from chromium c++ implementation code, not js.
    """
    file: str
    line: int

    def to_json(self):
        json = dict()
        json['file'] = self.file
        json['line'] = self.line
        return json

    @classmethod
    def from_json(cls, json):
        return cls(file=str(json['file']), line=int(json['line']))