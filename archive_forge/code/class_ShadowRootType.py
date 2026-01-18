from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
class ShadowRootType(enum.Enum):
    """
    Shadow root type.
    """
    USER_AGENT = 'user-agent'
    OPEN_ = 'open'
    CLOSED = 'closed'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)