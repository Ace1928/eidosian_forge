from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class RuleMatch:
    """
    Match data for a CSS rule.
    """
    rule: CSSRule
    matching_selectors: typing.List[int]

    def to_json(self):
        json = dict()
        json['rule'] = self.rule.to_json()
        json['matchingSelectors'] = [i for i in self.matching_selectors]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(rule=CSSRule.from_json(json['rule']), matching_selectors=[int(i) for i in json['matchingSelectors']])