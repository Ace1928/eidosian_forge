from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
@event_class('Security.securityStateChanged')
@dataclass
class SecurityStateChanged:
    """
    The security state of the page changed. No longer being sent.
    """
    security_state: SecurityState
    scheme_is_cryptographic: bool
    explanations: typing.List[SecurityStateExplanation]
    insecure_content_status: InsecureContentStatus
    summary: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SecurityStateChanged:
        return cls(security_state=SecurityState.from_json(json['securityState']), scheme_is_cryptographic=bool(json['schemeIsCryptographic']), explanations=[SecurityStateExplanation.from_json(i) for i in json['explanations']], insecure_content_status=InsecureContentStatus.from_json(json['insecureContentStatus']), summary=str(json['summary']) if 'summary' in json else None)