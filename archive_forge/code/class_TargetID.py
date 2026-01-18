from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
class TargetID(str):

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> TargetID:
        return cls(json)

    def __repr__(self):
        return 'TargetID({})'.format(super().__repr__())