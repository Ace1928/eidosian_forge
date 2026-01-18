from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class Specificity:
    """
    Specificity:
    https://drafts.csswg.org/selectors/#specificity-rules
    """
    a: int
    b: int
    c: int

    def to_json(self):
        json = dict()
        json['a'] = self.a
        json['b'] = self.b
        json['c'] = self.c
        return json

    @classmethod
    def from_json(cls, json):
        return cls(a=int(json['a']), b=int(json['b']), c=int(json['c']))