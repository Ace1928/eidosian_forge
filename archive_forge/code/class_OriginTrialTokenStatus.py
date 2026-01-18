from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
class OriginTrialTokenStatus(enum.Enum):
    """
    Origin Trial(https://www.chromium.org/blink/origin-trials) support.
    Status for an Origin Trial token.
    """
    SUCCESS = 'Success'
    NOT_SUPPORTED = 'NotSupported'
    INSECURE = 'Insecure'
    EXPIRED = 'Expired'
    WRONG_ORIGIN = 'WrongOrigin'
    INVALID_SIGNATURE = 'InvalidSignature'
    MALFORMED = 'Malformed'
    WRONG_VERSION = 'WrongVersion'
    FEATURE_DISABLED = 'FeatureDisabled'
    TOKEN_DISABLED = 'TokenDisabled'
    FEATURE_DISABLED_FOR_USER = 'FeatureDisabledForUser'
    UNKNOWN_TRIAL = 'UnknownTrial'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)