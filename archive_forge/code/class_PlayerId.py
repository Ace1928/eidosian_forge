from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class PlayerId(str):
    """
    Players will get an ID that is unique within the agent context.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> PlayerId:
        return cls(json)

    def __repr__(self):
        return 'PlayerId({})'.format(super().__repr__())