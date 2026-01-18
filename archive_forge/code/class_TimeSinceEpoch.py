from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
class TimeSinceEpoch(float):
    """
    UTC time in seconds, counted from January 1, 1970.
    """

    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> TimeSinceEpoch:
        return cls(json)

    def __repr__(self):
        return 'TimeSinceEpoch({})'.format(super().__repr__())