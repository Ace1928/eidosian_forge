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
class ReportingApiEndpoint:
    url: str
    group_name: str

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['groupName'] = self.group_name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(url=str(json['url']), group_name=str(json['groupName']))