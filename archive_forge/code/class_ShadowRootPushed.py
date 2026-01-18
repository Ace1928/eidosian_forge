from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.shadowRootPushed')
@dataclass
class ShadowRootPushed:
    """
    **EXPERIMENTAL**

    Called when shadow root is pushed into the element.
    """
    host_id: NodeId
    root: Node

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ShadowRootPushed:
        return cls(host_id=NodeId.from_json(json['hostId']), root=Node.from_json(json['root']))