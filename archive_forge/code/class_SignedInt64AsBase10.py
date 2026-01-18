from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
class SignedInt64AsBase10(str):

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> SignedInt64AsBase10:
        return cls(json)

    def __repr__(self):
        return 'SignedInt64AsBase10({})'.format(super().__repr__())