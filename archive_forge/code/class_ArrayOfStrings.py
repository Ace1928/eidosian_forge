from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page
class ArrayOfStrings(list):
    """
    Index of the string in the strings table.
    """

    def to_json(self) -> typing.List[StringIndex]:
        return self

    @classmethod
    def from_json(cls, json: typing.List[StringIndex]) -> ArrayOfStrings:
        return cls(json)

    def __repr__(self):
        return 'ArrayOfStrings({})'.format(super().__repr__())