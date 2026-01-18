from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class PromptDevice:
    """
    Device information displayed in a user prompt to select a device.
    """
    id_: DeviceId
    name: str

    def to_json(self):
        json = dict()
        json['id'] = self.id_.to_json()
        json['name'] = self.name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(id_=DeviceId.from_json(json['id']), name=str(json['name']))