from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
class AXValueSourceType(enum.Enum):
    """
    Enum of possible property sources.
    """
    ATTRIBUTE = 'attribute'
    IMPLICIT = 'implicit'
    STYLE = 'style'
    CONTENTS = 'contents'
    PLACEHOLDER = 'placeholder'
    RELATED_ELEMENT = 'relatedElement'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)