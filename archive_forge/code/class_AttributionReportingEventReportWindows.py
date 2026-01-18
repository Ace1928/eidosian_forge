from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class AttributionReportingEventReportWindows:
    start: int
    ends: typing.List[int]

    def to_json(self):
        json = dict()
        json['start'] = self.start
        json['ends'] = [i for i in self.ends]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(start=int(json['start']), ends=[int(i) for i in json['ends']])