from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@dataclass
class SamplingHeapProfile:
    """
    Sampling profile.
    """
    head: SamplingHeapProfileNode
    samples: typing.List[SamplingHeapProfileSample]

    def to_json(self):
        json = dict()
        json['head'] = self.head.to_json()
        json['samples'] = [i.to_json() for i in self.samples]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(head=SamplingHeapProfileNode.from_json(json['head']), samples=[SamplingHeapProfileSample.from_json(i) for i in json['samples']])