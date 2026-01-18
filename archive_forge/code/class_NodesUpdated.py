from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@event_class('Accessibility.nodesUpdated')
@dataclass
class NodesUpdated:
    """
    **EXPERIMENTAL**

    The nodesUpdated event is sent every time a previously requested node has changed the in tree.
    """
    nodes: typing.List[AXNode]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodesUpdated:
        return cls(nodes=[AXNode.from_json(i) for i in json['nodes']])