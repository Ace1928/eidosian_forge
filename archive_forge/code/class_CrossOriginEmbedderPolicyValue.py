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
class CrossOriginEmbedderPolicyValue(enum.Enum):
    NONE = 'None'
    CREDENTIALLESS = 'Credentialless'
    REQUIRE_CORP = 'RequireCorp'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)