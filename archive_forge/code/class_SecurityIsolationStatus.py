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
class SecurityIsolationStatus:
    coop: typing.Optional[CrossOriginOpenerPolicyStatus] = None
    coep: typing.Optional[CrossOriginEmbedderPolicyStatus] = None
    csp: typing.Optional[typing.List[ContentSecurityPolicyStatus]] = None

    def to_json(self):
        json = dict()
        if self.coop is not None:
            json['coop'] = self.coop.to_json()
        if self.coep is not None:
            json['coep'] = self.coep.to_json()
        if self.csp is not None:
            json['csp'] = [i.to_json() for i in self.csp]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(coop=CrossOriginOpenerPolicyStatus.from_json(json['coop']) if 'coop' in json else None, coep=CrossOriginEmbedderPolicyStatus.from_json(json['coep']) if 'coep' in json else None, csp=[ContentSecurityPolicyStatus.from_json(i) for i in json['csp']] if 'csp' in json else None)