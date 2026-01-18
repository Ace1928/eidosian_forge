from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class ExecutionContextId(int):
    """
    Id of an execution context.
    """

    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> ExecutionContextId:
        return cls(json)

    def __repr__(self):
        return 'ExecutionContextId({})'.format(super().__repr__())