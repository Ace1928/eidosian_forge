from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
class HeapSnapshotObjectId(str):
    """
    Heap snapshot object id.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> HeapSnapshotObjectId:
        return cls(json)

    def __repr__(self):
        return 'HeapSnapshotObjectId({})'.format(super().__repr__())