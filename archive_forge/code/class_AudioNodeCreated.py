from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAudio.audioNodeCreated')
@dataclass
class AudioNodeCreated:
    """
    Notifies that a new AudioNode has been created.
    """
    node: AudioNode

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioNodeCreated:
        return cls(node=AudioNode.from_json(json['node']))