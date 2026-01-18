from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
class MixedContentResolutionStatus(enum.Enum):
    MIXED_CONTENT_BLOCKED = 'MixedContentBlocked'
    MIXED_CONTENT_AUTOMATICALLY_UPGRADED = 'MixedContentAutomaticallyUpgraded'
    MIXED_CONTENT_WARNING = 'MixedContentWarning'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)