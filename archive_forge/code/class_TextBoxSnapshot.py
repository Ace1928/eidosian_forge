from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page
@dataclass
class TextBoxSnapshot:
    """
    Table of details of the post layout rendered text positions. The exact layout should not be regarded as
    stable and may change between versions.
    """
    layout_index: typing.List[int]
    bounds: typing.List[Rectangle]
    start: typing.List[int]
    length: typing.List[int]

    def to_json(self):
        json = dict()
        json['layoutIndex'] = [i for i in self.layout_index]
        json['bounds'] = [i.to_json() for i in self.bounds]
        json['start'] = [i for i in self.start]
        json['length'] = [i for i in self.length]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(layout_index=[int(i) for i in json['layoutIndex']], bounds=[Rectangle.from_json(i) for i in json['bounds']], start=[int(i) for i in json['start']], length=[int(i) for i in json['length']])