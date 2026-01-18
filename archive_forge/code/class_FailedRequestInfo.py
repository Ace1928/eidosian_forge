from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
@dataclass
class FailedRequestInfo:
    url: str
    failure_message: str
    request_id: typing.Optional[network.RequestId] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['failureMessage'] = self.failure_message
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(url=str(json['url']), failure_message=str(json['failureMessage']), request_id=network.RequestId.from_json(json['requestId']) if 'requestId' in json else None)