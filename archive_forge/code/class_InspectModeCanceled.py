from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@event_class('Overlay.inspectModeCanceled')
@dataclass
class InspectModeCanceled:
    """
    Fired when user cancels the inspect mode.
    """

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InspectModeCanceled:
        return cls()