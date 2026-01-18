from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
class DOMBreakpointType(enum.Enum):
    """
    DOM breakpoint type.
    """
    SUBTREE_MODIFIED = 'subtree-modified'
    ATTRIBUTE_MODIFIED = 'attribute-modified'
    NODE_REMOVED = 'node-removed'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)