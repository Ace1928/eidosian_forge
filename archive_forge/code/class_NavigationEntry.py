from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@dataclass
class NavigationEntry:
    """
    Navigation history entry.
    """
    id_: int
    url: str
    user_typed_url: str
    title: str
    transition_type: TransitionType

    def to_json(self):
        json = dict()
        json['id'] = self.id_
        json['url'] = self.url
        json['userTypedURL'] = self.user_typed_url
        json['title'] = self.title
        json['transitionType'] = self.transition_type.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(id_=int(json['id']), url=str(json['url']), user_typed_url=str(json['userTypedURL']), title=str(json['title']), transition_type=TransitionType.from_json(json['transitionType']))