from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
from . import network
from . import page
class RequestStage(enum.Enum):
    """
    Stages of the request to handle. Request will intercept before the request is
    sent. Response will intercept after the response is received (but before response
    body is received.
    """
    REQUEST = 'Request'
    RESPONSE = 'Response'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)