from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
@dataclass
class FunctionCoverage:
    """
    Coverage data for a JavaScript function.
    """
    function_name: str
    ranges: typing.List[CoverageRange]
    is_block_coverage: bool

    def to_json(self):
        json = dict()
        json['functionName'] = self.function_name
        json['ranges'] = [i.to_json() for i in self.ranges]
        json['isBlockCoverage'] = self.is_block_coverage
        return json

    @classmethod
    def from_json(cls, json):
        return cls(function_name=str(json['functionName']), ranges=[CoverageRange.from_json(i) for i in json['ranges']], is_block_coverage=bool(json['isBlockCoverage']))