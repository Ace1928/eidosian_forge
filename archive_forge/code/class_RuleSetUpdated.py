from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@event_class('Preload.ruleSetUpdated')
@dataclass
class RuleSetUpdated:
    """
    Upsert. Currently, it is only emitted when a rule set added.
    """
    rule_set: RuleSet

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RuleSetUpdated:
        return cls(rule_set=RuleSet.from_json(json['ruleSet']))