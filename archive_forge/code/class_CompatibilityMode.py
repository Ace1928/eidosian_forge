from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
class CompatibilityMode(enum.Enum):
    """
    Document compatibility mode.
    """
    QUIRKS_MODE = 'QuirksMode'
    LIMITED_QUIRKS_MODE = 'LimitedQuirksMode'
    NO_QUIRKS_MODE = 'NoQuirksMode'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)