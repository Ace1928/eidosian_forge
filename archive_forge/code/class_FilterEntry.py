from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
@dataclass
class FilterEntry:
    """
    A filter used by target query/discovery/auto-attach operations.
    """
    exclude: typing.Optional[bool] = None
    type_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.exclude is not None:
            json['exclude'] = self.exclude
        if self.type_ is not None:
            json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(exclude=bool(json['exclude']) if 'exclude' in json else None, type_=str(json['type']) if 'type' in json else None)