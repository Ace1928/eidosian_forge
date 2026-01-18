from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@dataclass
class AppManifestParsedProperties:
    """
    Parsed app manifest properties.
    """
    scope: str

    def to_json(self):
        json = dict()
        json['scope'] = self.scope
        return json

    @classmethod
    def from_json(cls, json):
        return cls(scope=str(json['scope']))