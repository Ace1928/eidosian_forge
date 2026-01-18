from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
class AXValueNativeSourceType(enum.Enum):
    """
    Enum of possible native property sources (as a subtype of a particular AXValueSourceType).
    """
    FIGCAPTION = 'figcaption'
    LABEL = 'label'
    LABELFOR = 'labelfor'
    LABELWRAPPED = 'labelwrapped'
    LEGEND = 'legend'
    TABLECAPTION = 'tablecaption'
    TITLE = 'title'
    OTHER = 'other'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)