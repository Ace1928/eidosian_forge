from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSPositionFallbackRule:
    """
    CSS position-fallback rule representation.
    """
    name: Value
    try_rules: typing.List[CSSTryRule]

    def to_json(self):
        json = dict()
        json['name'] = self.name.to_json()
        json['tryRules'] = [i.to_json() for i in self.try_rules]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=Value.from_json(json['name']), try_rules=[CSSTryRule.from_json(i) for i in json['tryRules']])