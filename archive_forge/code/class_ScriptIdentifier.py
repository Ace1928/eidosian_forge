from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
class ScriptIdentifier(str):
    """
    Unique script identifier.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ScriptIdentifier:
        return cls(json)

    def __repr__(self):
        return 'ScriptIdentifier({})'.format(super().__repr__())