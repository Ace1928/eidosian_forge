from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.pseudoElementAdded')
@dataclass
class PseudoElementAdded:
    """
    **EXPERIMENTAL**

    Called when a pseudo element is added to an element.
    """
    parent_id: NodeId
    pseudo_element: Node

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PseudoElementAdded:
        return cls(parent_id=NodeId.from_json(json['parentId']), pseudo_element=Node.from_json(json['pseudoElement']))