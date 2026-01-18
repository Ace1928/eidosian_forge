from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
class CSPViolationType(enum.Enum):
    """
    CSP Violation type.
    """
    TRUSTEDTYPE_SINK_VIOLATION = 'trustedtype-sink-violation'
    TRUSTEDTYPE_POLICY_VIOLATION = 'trustedtype-policy-violation'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)