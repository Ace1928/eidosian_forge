from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
class StreamCompression(enum.Enum):
    """
    Compression type to use for traces returned via streams.
    """
    NONE = 'none'
    GZIP = 'gzip'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)