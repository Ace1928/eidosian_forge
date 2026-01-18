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
class ServiceWorkerResponseSource(enum.Enum):
    """
    Source of serviceworker response.
    """
    CACHE_STORAGE = 'cache-storage'
    HTTP_CACHE = 'http-cache'
    FALLBACK_CODE = 'fallback-code'
    NETWORK = 'network'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)