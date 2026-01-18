from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
class BrowserContextID(str):

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> BrowserContextID:
        return cls(json)

    def __repr__(self):
        return 'BrowserContextID({})'.format(super().__repr__())