from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@dataclass
class SamplingHeapProfileNode:
    """
    Sampling Heap Profile node. Holds callsite information, allocation statistics and child nodes.
    """
    call_frame: runtime.CallFrame
    self_size: float
    id_: int
    children: typing.List[SamplingHeapProfileNode]

    def to_json(self):
        json = dict()
        json['callFrame'] = self.call_frame.to_json()
        json['selfSize'] = self.self_size
        json['id'] = self.id_
        json['children'] = [i.to_json() for i in self.children]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(call_frame=runtime.CallFrame.from_json(json['callFrame']), self_size=float(json['selfSize']), id_=int(json['id']), children=[SamplingHeapProfileNode.from_json(i) for i in json['children']])