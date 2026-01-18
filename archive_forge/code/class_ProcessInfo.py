from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class ProcessInfo:
    """
    Represents process info.
    """
    type_: str
    id_: int
    cpu_time: float

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['id'] = self.id_
        json['cpuTime'] = self.cpu_time
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=str(json['type']), id_=int(json['id']), cpu_time=float(json['cpuTime']))