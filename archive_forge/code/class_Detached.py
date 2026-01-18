from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Inspector.detached')
@dataclass
class Detached:
    """
    Fired when remote debugging connection is about to be terminated. Contains detach reason.
    """
    reason: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Detached:
        return cls(reason=str(json['reason']))