from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
class FillingStrategy(enum.Enum):
    """
    Specified whether a filled field was done so by using the html autocomplete attribute or autofill heuristics.
    """
    AUTOCOMPLETE_ATTRIBUTE = 'autocompleteAttribute'
    AUTOFILL_INFERRED = 'autofillInferred'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)