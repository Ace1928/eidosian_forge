from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
@event_class('Target.targetInfoChanged')
@dataclass
class TargetInfoChanged:
    """
    Issued when some information about a target has changed. This only happens between
    ``targetCreated`` and ``targetDestroyed``.
    """
    target_info: TargetInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetInfoChanged:
        return cls(target_info=TargetInfo.from_json(json['targetInfo']))