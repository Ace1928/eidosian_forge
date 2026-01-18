from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
class HeavyAdReason(enum.Enum):
    NETWORK_TOTAL_LIMIT = 'NetworkTotalLimit'
    CPU_TOTAL_LIMIT = 'CpuTotalLimit'
    CPU_PEAK_LIMIT = 'CpuPeakLimit'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)