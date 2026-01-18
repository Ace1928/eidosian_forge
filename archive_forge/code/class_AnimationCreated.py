from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@event_class('Animation.animationCreated')
@dataclass
class AnimationCreated:
    """
    Event for each animation that has been created.
    """
    id_: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationCreated:
        return cls(id_=str(json['id']))