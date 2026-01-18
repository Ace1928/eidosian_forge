from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class InheritedStyleEntry:
    """
    Inherited CSS rule collection from ancestor node.
    """
    matched_css_rules: typing.List[RuleMatch]
    inline_style: typing.Optional[CSSStyle] = None

    def to_json(self):
        json = dict()
        json['matchedCSSRules'] = [i.to_json() for i in self.matched_css_rules]
        if self.inline_style is not None:
            json['inlineStyle'] = self.inline_style.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(matched_css_rules=[RuleMatch.from_json(i) for i in json['matchedCSSRules']], inline_style=CSSStyle.from_json(json['inlineStyle']) if 'inlineStyle' in json else None)