from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@dataclass
class DebugSymbols:
    """
    Debug symbols available for a wasm script.
    """
    type_: str
    external_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.external_url is not None:
            json['externalURL'] = self.external_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=str(json['type']), external_url=str(json['externalURL']) if 'externalURL' in json else None)