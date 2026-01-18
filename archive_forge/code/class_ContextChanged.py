from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAudio.contextChanged')
@dataclass
class ContextChanged:
    """
    Notifies that existing BaseAudioContext has changed some properties (id stays the same)..
    """
    context: BaseAudioContext

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextChanged:
        return cls(context=BaseAudioContext.from_json(json['context']))