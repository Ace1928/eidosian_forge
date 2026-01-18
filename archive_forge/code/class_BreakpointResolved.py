from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@event_class('Debugger.breakpointResolved')
@dataclass
class BreakpointResolved:
    """
    Fired when breakpoint is resolved to an actual script and location.
    """
    breakpoint_id: BreakpointId
    location: Location

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BreakpointResolved:
        return cls(breakpoint_id=BreakpointId.from_json(json['breakpointId']), location=Location.from_json(json['location']))