from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@dataclass
class KeyPath:
    """
    Key path.
    """
    type_: str
    string: typing.Optional[str] = None
    array: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.string is not None:
            json['string'] = self.string
        if self.array is not None:
            json['array'] = [i for i in self.array]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=str(json['type']), string=str(json['string']) if 'string' in json else None, array=[str(i) for i in json['array']] if 'array' in json else None)