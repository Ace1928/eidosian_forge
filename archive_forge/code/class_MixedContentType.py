from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
class MixedContentType(enum.Enum):
    """
    A description of mixed content (HTTP resources on HTTPS pages), as defined by
    https://www.w3.org/TR/mixed-content/#categories
    """
    BLOCKABLE = 'blockable'
    OPTIONALLY_BLOCKABLE = 'optionally-blockable'
    NONE = 'none'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)