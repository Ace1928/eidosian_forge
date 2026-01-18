from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
@event_class('Target.targetCrashed')
@dataclass
class TargetCrashed:
    """
    Issued when a target has crashed.
    """
    target_id: TargetID
    status: str
    error_code: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetCrashed:
        return cls(target_id=TargetID.from_json(json['targetId']), status=str(json['status']), error_code=int(json['errorCode']))