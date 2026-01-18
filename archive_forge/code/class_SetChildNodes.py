from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.setChildNodes')
@dataclass
class SetChildNodes:
    """
    Fired when backend wants to provide client with the missing DOM structure. This happens upon
    most of the calls requesting node ids.
    """
    parent_id: NodeId
    nodes: typing.List[Node]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SetChildNodes:
        return cls(parent_id=NodeId.from_json(json['parentId']), nodes=[Node.from_json(i) for i in json['nodes']])