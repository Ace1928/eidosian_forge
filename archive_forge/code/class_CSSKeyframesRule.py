from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSKeyframesRule:
    """
    CSS keyframes rule representation.
    """
    animation_name: Value
    keyframes: typing.List[CSSKeyframeRule]

    def to_json(self):
        json = dict()
        json['animationName'] = self.animation_name.to_json()
        json['keyframes'] = [i.to_json() for i in self.keyframes]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(animation_name=Value.from_json(json['animationName']), keyframes=[CSSKeyframeRule.from_json(i) for i in json['keyframes']])