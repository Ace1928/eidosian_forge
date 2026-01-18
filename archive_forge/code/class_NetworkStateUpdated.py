from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
@event_class('ApplicationCache.networkStateUpdated')
@dataclass
class NetworkStateUpdated:
    is_now_online: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NetworkStateUpdated:
        return cls(is_now_online=bool(json['isNowOnline']))