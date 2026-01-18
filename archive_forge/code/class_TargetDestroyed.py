from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
@event_class('Target.targetDestroyed')
@dataclass
class TargetDestroyed:
    """
    Issued when a target is destroyed.
    """
    target_id: TargetID

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetDestroyed:
        return cls(target_id=TargetID.from_json(json['targetId']))