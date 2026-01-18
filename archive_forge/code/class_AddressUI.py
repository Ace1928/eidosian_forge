from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class AddressUI:
    """
    Defines how an address can be displayed like in chrome://settings/addresses.
    Address UI is a two dimensional array, each inner array is an "address information line", and when rendered in a UI surface should be displayed as such.
    The following address UI for instance:
    [[{name: "GIVE_NAME", value: "Jon"}, {name: "FAMILY_NAME", value: "Doe"}], [{name: "CITY", value: "Munich"}, {name: "ZIP", value: "81456"}]]
    should allow the receiver to render:
    Jon Doe
    Munich 81456
    """
    address_fields: typing.List[AddressFields]

    def to_json(self):
        json = dict()
        json['addressFields'] = [i.to_json() for i in self.address_fields]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(address_fields=[AddressFields.from_json(i) for i in json['addressFields']])