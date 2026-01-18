from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class ScriptId(str):
    """
    Unique script identifier.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ScriptId:
        return cls(json)

    def __repr__(self):
        return 'ScriptId({})'.format(super().__repr__())