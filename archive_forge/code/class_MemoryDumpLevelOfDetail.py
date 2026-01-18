from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
class MemoryDumpLevelOfDetail(enum.Enum):
    """
    Details exposed when memory request explicitly declared.
    Keep consistent with memory_dump_request_args.h and
    memory_instrumentation.mojom
    """
    BACKGROUND = 'background'
    LIGHT = 'light'
    DETAILED = 'detailed'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)