from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
@dataclass
class RemoteLocation:
    host: str
    port: int

    def to_json(self):
        json = dict()
        json['host'] = self.host
        json['port'] = self.port
        return json

    @classmethod
    def from_json(cls, json):
        return cls(host=str(json['host']), port=int(json['port']))