from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class ChannelCountMode(enum.Enum):
    """
    Enum of AudioNode::ChannelCountMode from the spec
    """
    CLAMPED_MAX = 'clamped-max'
    EXPLICIT = 'explicit'
    MAX_ = 'max'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)