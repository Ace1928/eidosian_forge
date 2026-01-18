from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class PrivatePropertyDescriptor:
    """
    Object private field descriptor.
    """
    name: str
    value: typing.Optional[RemoteObject] = None
    get: typing.Optional[RemoteObject] = None
    set_: typing.Optional[RemoteObject] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.get is not None:
            json['get'] = self.get.to_json()
        if self.set_ is not None:
            json['set'] = self.set_.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), value=RemoteObject.from_json(json['value']) if 'value' in json else None, get=RemoteObject.from_json(json['get']) if 'get' in json else None, set_=RemoteObject.from_json(json['set']) if 'set' in json else None)