from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class AddressFields:
    """
    A list of address fields.
    """
    fields: typing.List[AddressField]

    def to_json(self):
        json = dict()
        json['fields'] = [i.to_json() for i in self.fields]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(fields=[AddressField.from_json(i) for i in json['fields']])