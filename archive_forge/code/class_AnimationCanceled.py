from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@event_class('Animation.animationCanceled')
@dataclass
class AnimationCanceled:
    """
    Event for when an animation has been cancelled.
    """
    id_: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationCanceled:
        return cls(id_=str(json['id']))