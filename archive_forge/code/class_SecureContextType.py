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
class SecureContextType(enum.Enum):
    """
    Indicates whether the frame is a secure context and why it is the case.
    """
    SECURE = 'Secure'
    SECURE_LOCALHOST = 'SecureLocalhost'
    INSECURE_SCHEME = 'InsecureScheme'
    INSECURE_ANCESTOR = 'InsecureAncestor'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)