from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class DeviceId(str):
    """
    A device id.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> DeviceId:
        return cls(json)

    def __repr__(self):
        return 'DeviceId({})'.format(super().__repr__())