from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
class RegistrationID(str):

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RegistrationID:
        return cls(json)

    def __repr__(self):
        return 'RegistrationID({})'.format(super().__repr__())