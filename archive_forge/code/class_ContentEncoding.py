from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
class ContentEncoding(enum.Enum):
    """
    List of content encodings supported by the backend.
    """
    DEFLATE = 'deflate'
    GZIP = 'gzip'
    BR = 'br'
    ZSTD = 'zstd'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)