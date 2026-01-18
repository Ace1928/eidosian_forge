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
class AlternateProtocolUsage(enum.Enum):
    """
    The reason why Chrome uses a specific transport protocol for HTTP semantics.
    """
    ALTERNATIVE_JOB_WON_WITHOUT_RACE = 'alternativeJobWonWithoutRace'
    ALTERNATIVE_JOB_WON_RACE = 'alternativeJobWonRace'
    MAIN_JOB_WON_RACE = 'mainJobWonRace'
    MAPPING_MISSING = 'mappingMissing'
    BROKEN = 'broken'
    DNS_ALPN_H3_JOB_WON_WITHOUT_RACE = 'dnsAlpnH3JobWonWithoutRace'
    DNS_ALPN_H3_JOB_WON_RACE = 'dnsAlpnH3JobWonRace'
    UNSPECIFIED_REASON = 'unspecifiedReason'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)