from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
class DisabledImageType(enum.Enum):
    """
    Enum of image types that can be disabled.
    """
    AVIF = 'avif'
    WEBP = 'webp'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)