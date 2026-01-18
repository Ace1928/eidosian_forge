from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@dataclass
class SamplingHeapProfileSample:
    """
    A single sample from a sampling profile.
    """
    size: float
    node_id: int
    ordinal: float

    def to_json(self):
        json = dict()
        json['size'] = self.size
        json['nodeId'] = self.node_id
        json['ordinal'] = self.ordinal
        return json

    @classmethod
    def from_json(cls, json):
        return cls(size=float(json['size']), node_id=int(json['nodeId']), ordinal=float(json['ordinal']))