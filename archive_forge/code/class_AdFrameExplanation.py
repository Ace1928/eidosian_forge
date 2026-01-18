from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
class AdFrameExplanation(enum.Enum):
    PARENT_IS_AD = 'ParentIsAd'
    CREATED_BY_AD_SCRIPT = 'CreatedByAdScript'
    MATCHED_BLOCKING_RULE = 'MatchedBlockingRule'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)