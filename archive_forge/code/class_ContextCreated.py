from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAudio.contextCreated')
@dataclass
class ContextCreated:
    """
    Notifies that a new BaseAudioContext has been created.
    """
    context: BaseAudioContext

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextCreated:
        return cls(context=BaseAudioContext.from_json(json['context']))