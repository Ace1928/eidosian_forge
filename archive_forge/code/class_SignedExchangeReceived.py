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
@event_class('Network.signedExchangeReceived')
@dataclass
class SignedExchangeReceived:
    """
    **EXPERIMENTAL**

    Fired when a signed exchange was received over the network
    """
    request_id: RequestId
    info: SignedExchangeInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SignedExchangeReceived:
        return cls(request_id=RequestId.from_json(json['requestId']), info=SignedExchangeInfo.from_json(json['info']))