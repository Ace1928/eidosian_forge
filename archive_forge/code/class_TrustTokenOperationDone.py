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
@event_class('Network.trustTokenOperationDone')
@dataclass
class TrustTokenOperationDone:
    """
    **EXPERIMENTAL**

    Fired exactly once for each Trust Token operation. Depending on
    the type of the operation and whether the operation succeeded or
    failed, the event is fired before the corresponding request was sent
    or after the response was received.
    """
    status: str
    type_: TrustTokenOperationType
    request_id: RequestId
    top_level_origin: typing.Optional[str]
    issuer_origin: typing.Optional[str]
    issued_token_count: typing.Optional[int]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TrustTokenOperationDone:
        return cls(status=str(json['status']), type_=TrustTokenOperationType.from_json(json['type']), request_id=RequestId.from_json(json['requestId']), top_level_origin=str(json['topLevelOrigin']) if 'topLevelOrigin' in json else None, issuer_origin=str(json['issuerOrigin']) if 'issuerOrigin' in json else None, issued_token_count=int(json['issuedTokenCount']) if 'issuedTokenCount' in json else None)