from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@dataclass
class WasmDisassemblyChunk:
    lines: typing.List[str]
    bytecode_offsets: typing.List[int]

    def to_json(self):
        json = dict()
        json['lines'] = [i for i in self.lines]
        json['bytecodeOffsets'] = [i for i in self.bytecode_offsets]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(lines=[str(i) for i in json['lines']], bytecode_offsets=[int(i) for i in json['bytecodeOffsets']])