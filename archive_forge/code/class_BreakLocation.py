from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@dataclass
class BreakLocation:
    script_id: runtime.ScriptId
    line_number: int
    column_number: typing.Optional[int] = None
    type_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['lineNumber'] = self.line_number
        if self.column_number is not None:
            json['columnNumber'] = self.column_number
        if self.type_ is not None:
            json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(script_id=runtime.ScriptId.from_json(json['scriptId']), line_number=int(json['lineNumber']), column_number=int(json['columnNumber']) if 'columnNumber' in json else None, type_=str(json['type']) if 'type' in json else None)