from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.attributeModified')
@dataclass
class AttributeModified:
    """
    Fired when ``Element``'s attribute is modified.
    """
    node_id: NodeId
    name: str
    value: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttributeModified:
        return cls(node_id=NodeId.from_json(json['nodeId']), name=str(json['name']), value=str(json['value']))