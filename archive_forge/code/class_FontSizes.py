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
class FontSizes:
    """
    Default font sizes.
    """
    standard: typing.Optional[int] = None
    fixed: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        if self.standard is not None:
            json['standard'] = self.standard
        if self.fixed is not None:
            json['fixed'] = self.fixed
        return json

    @classmethod
    def from_json(cls, json):
        return cls(standard=int(json['standard']) if 'standard' in json else None, fixed=int(json['fixed']) if 'fixed' in json else None)