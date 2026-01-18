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
class MonotonicTime(float):
    """
    Monotonically increasing time in seconds since an arbitrary point in the past.
    """

    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> MonotonicTime:
        return cls(json)

    def __repr__(self):
        return 'MonotonicTime({})'.format(super().__repr__())