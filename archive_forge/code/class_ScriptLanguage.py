from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
class ScriptLanguage(enum.Enum):
    """
    Enum of possible script languages.
    """
    JAVA_SCRIPT = 'JavaScript'
    WEB_ASSEMBLY = 'WebAssembly'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)