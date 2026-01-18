from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSPropertyRegistration:
    """
    Representation of a custom property registration through CSS.registerProperty
    """
    property_name: str
    inherits: bool
    syntax: str
    initial_value: typing.Optional[Value] = None

    def to_json(self):
        json = dict()
        json['propertyName'] = self.property_name
        json['inherits'] = self.inherits
        json['syntax'] = self.syntax
        if self.initial_value is not None:
            json['initialValue'] = self.initial_value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(property_name=str(json['propertyName']), inherits=bool(json['inherits']), syntax=str(json['syntax']), initial_value=Value.from_json(json['initialValue']) if 'initialValue' in json else None)