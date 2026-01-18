from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class SamplingProfileNode:
    """
    Heap profile sample.
    """
    size: float
    total: float
    stack: typing.List[str]

    def to_json(self):
        json = dict()
        json['size'] = self.size
        json['total'] = self.total
        json['stack'] = [i for i in self.stack]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(size=float(json['size']), total=float(json['total']), stack=[str(i) for i in json['stack']])