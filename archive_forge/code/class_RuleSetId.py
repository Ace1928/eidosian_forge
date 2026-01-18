from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
class RuleSetId(str):
    """
    Unique id
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RuleSetId:
        return cls(json)

    def __repr__(self):
        return 'RuleSetId({})'.format(super().__repr__())