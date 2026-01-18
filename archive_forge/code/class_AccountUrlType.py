from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class AccountUrlType(enum.Enum):
    """
    The URLs that each account has
    """
    TERMS_OF_SERVICE = 'TermsOfService'
    PRIVACY_POLICY = 'PrivacyPolicy'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)