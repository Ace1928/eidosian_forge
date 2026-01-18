from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
class CertificateErrorAction(enum.Enum):
    """
    The action to take when a certificate error occurs. continue will continue processing the
    request and cancel will cancel the request.
    """
    CONTINUE = 'continue'
    CANCEL = 'cancel'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)