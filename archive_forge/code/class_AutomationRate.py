from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class AutomationRate(enum.Enum):
    """
    Enum of AudioParam::AutomationRate from the spec
    """
    A_RATE = 'a-rate'
    K_RATE = 'k-rate'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)