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
class RequestId(str):
    """
    Unique request identifier.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RequestId:
        return cls(json)

    def __repr__(self):
        return 'RequestId({})'.format(super().__repr__())