from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.childNodeRemoved')
@dataclass
class ChildNodeRemoved:
    """
    Mirrors ``DOMNodeRemoved`` event.
    """
    parent_node_id: NodeId
    node_id: NodeId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ChildNodeRemoved:
        return cls(parent_node_id=NodeId.from_json(json['parentNodeId']), node_id=NodeId.from_json(json['nodeId']))