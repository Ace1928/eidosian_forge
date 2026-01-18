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
@dataclass
class TrustTokenParams:
    """
    Determines what type of Trust Token operation is executed and
    depending on the type, some additional parameters. The values
    are specified in third_party/blink/renderer/core/fetch/trust_token.idl.
    """
    operation: TrustTokenOperationType
    refresh_policy: str
    issuers: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['operation'] = self.operation.to_json()
        json['refreshPolicy'] = self.refresh_policy
        if self.issuers is not None:
            json['issuers'] = [i for i in self.issuers]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(operation=TrustTokenOperationType.from_json(json['operation']), refresh_policy=str(json['refreshPolicy']), issuers=[str(i) for i in json['issuers']] if 'issuers' in json else None)