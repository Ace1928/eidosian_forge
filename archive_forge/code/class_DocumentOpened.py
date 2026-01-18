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
@event_class('Page.documentOpened')
@dataclass
class DocumentOpened:
    """
    **EXPERIMENTAL**

    Fired when opening document to write to.
    """
    frame: Frame

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DocumentOpened:
        return cls(frame=Frame.from_json(json['frame']))