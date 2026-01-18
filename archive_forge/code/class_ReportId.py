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
class ReportId(str):

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ReportId:
        return cls(json)

    def __repr__(self):
        return 'ReportId({})'.format(super().__repr__())