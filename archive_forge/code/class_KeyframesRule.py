from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@dataclass
class KeyframesRule:
    """
    Keyframes Rule
    """
    keyframes: typing.List[KeyframeStyle]
    name: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['keyframes'] = [i.to_json() for i in self.keyframes]
        if self.name is not None:
            json['name'] = self.name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(keyframes=[KeyframeStyle.from_json(i) for i in json['keyframes']], name=str(json['name']) if 'name' in json else None)