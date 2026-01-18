from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class PlayerError:
    """
    Corresponds to kMediaError
    """
    type_: str
    error_code: str

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['errorCode'] = self.error_code
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=str(json['type']), error_code=str(json['errorCode']))