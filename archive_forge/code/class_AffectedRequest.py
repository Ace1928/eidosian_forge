from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
@dataclass
class AffectedRequest:
    """
    Information about a request that is affected by an inspector issue.
    """
    request_id: network.RequestId
    url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['requestId'] = self.request_id.to_json()
        if self.url is not None:
            json['url'] = self.url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(request_id=network.RequestId.from_json(json['requestId']), url=str(json['url']) if 'url' in json else None)