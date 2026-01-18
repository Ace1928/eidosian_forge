from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page
@dataclass
class ComputedStyle:
    """
    A subset of the full ComputedStyle as defined by the request whitelist.
    """
    properties: typing.List[NameValue]

    def to_json(self):
        json = dict()
        json['properties'] = [i.to_json() for i in self.properties]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(properties=[NameValue.from_json(i) for i in json['properties']])