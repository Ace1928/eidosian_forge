from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class AuthenticatorProtocol(enum.Enum):
    U2F = 'u2f'
    CTAP2 = 'ctap2'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)