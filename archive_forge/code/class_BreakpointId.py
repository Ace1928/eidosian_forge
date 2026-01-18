from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
class BreakpointId(str):
    """
    Breakpoint identifier.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> BreakpointId:
        return cls(json)

    def __repr__(self):
        return 'BreakpointId({})'.format(super().__repr__())