from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Inspector.targetReloadedAfterCrash')
@dataclass
class TargetReloadedAfterCrash:
    """
    Fired when debugging target has reloaded after crash
    """

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetReloadedAfterCrash:
        return cls()