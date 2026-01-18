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
class ServiceWorkerRouterSource(enum.Enum):
    """
    Source of service worker router.
    """
    NETWORK = 'network'
    CACHE = 'cache'
    FETCH_EVENT = 'fetch-event'
    RACE_NETWORK_AND_FETCH_HANDLER = 'race-network-and-fetch-handler'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)