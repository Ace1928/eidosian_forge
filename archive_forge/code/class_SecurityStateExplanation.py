from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
@dataclass
class SecurityStateExplanation:
    """
    An explanation of an factor contributing to the security state.
    """
    security_state: SecurityState
    title: str
    summary: str
    description: str
    mixed_content_type: MixedContentType
    certificate: typing.List[str]
    recommendations: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['securityState'] = self.security_state.to_json()
        json['title'] = self.title
        json['summary'] = self.summary
        json['description'] = self.description
        json['mixedContentType'] = self.mixed_content_type.to_json()
        json['certificate'] = [i for i in self.certificate]
        if self.recommendations is not None:
            json['recommendations'] = [i for i in self.recommendations]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(security_state=SecurityState.from_json(json['securityState']), title=str(json['title']), summary=str(json['summary']), description=str(json['description']), mixed_content_type=MixedContentType.from_json(json['mixedContentType']), certificate=[str(i) for i in json['certificate']], recommendations=[str(i) for i in json['recommendations']] if 'recommendations' in json else None)