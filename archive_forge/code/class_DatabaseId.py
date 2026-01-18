from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class DatabaseId(str):
    """
    Unique identifier of Database object.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> DatabaseId:
        return cls(json)

    def __repr__(self):
        return 'DatabaseId({})'.format(super().__repr__())