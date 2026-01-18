from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.inlineStyleInvalidated')
@dataclass
class InlineStyleInvalidated:
    """
    **EXPERIMENTAL**

    Fired when ``Element``'s inline style is modified via a CSS property modification.
    """
    node_ids: typing.List[NodeId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InlineStyleInvalidated:
        return cls(node_ids=[NodeId.from_json(i) for i in json['nodeIds']])