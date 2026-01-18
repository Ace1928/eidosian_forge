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
class NavigatorUserAgentIssueDetails:
    url: str
    location: typing.Optional[SourceCodeLocation] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        if self.location is not None:
            json['location'] = self.location.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(url=str(json['url']), location=SourceCodeLocation.from_json(json['location']) if 'location' in json else None)