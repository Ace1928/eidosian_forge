from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@dataclass
class ObjectStoreIndex:
    """
    Object store index.
    """
    name: str
    key_path: KeyPath
    unique: bool
    multi_entry: bool

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['keyPath'] = self.key_path.to_json()
        json['unique'] = self.unique
        json['multiEntry'] = self.multi_entry
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), key_path=KeyPath.from_json(json['keyPath']), unique=bool(json['unique']), multi_entry=bool(json['multiEntry']))