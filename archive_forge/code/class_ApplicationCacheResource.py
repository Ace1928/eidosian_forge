from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
@dataclass
class ApplicationCacheResource:
    """
    Detailed application cache resource information.
    """
    url: str
    size: int
    type_: str

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['size'] = self.size
        json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(url=str(json['url']), size=int(json['size']), type_=str(json['type']))