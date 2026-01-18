from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.documentUpdated')
@dataclass
class DocumentUpdated:
    """
    Fired when ``Document`` has been totally updated. Node ids are no longer valid.
    """

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DocumentUpdated:
        return cls()