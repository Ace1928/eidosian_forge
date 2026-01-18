from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
class PaintProfile(list):
    """
    Array of timings, one per paint step.
    """

    def to_json(self) -> typing.List[float]:
        return self

    @classmethod
    def from_json(cls, json: typing.List[float]) -> PaintProfile:
        return cls(json)

    def __repr__(self):
        return 'PaintProfile({})'.format(super().__repr__())