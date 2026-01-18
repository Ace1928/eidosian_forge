from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
class BackendNodeId(int):
    """
    Unique DOM node identifier used to reference a node that may not have been pushed to the
    front-end.
    """

    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> BackendNodeId:
        return cls(json)

    def __repr__(self):
        return 'BackendNodeId({})'.format(super().__repr__())