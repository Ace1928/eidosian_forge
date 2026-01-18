from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@event_class('Accessibility.loadComplete')
@dataclass
class LoadComplete:
    """
    **EXPERIMENTAL**

    The loadComplete event mirrors the load complete event sent by the browser to assistive
    technology when the web page has finished loading.
    """
    root: AXNode

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadComplete:
        return cls(root=AXNode.from_json(json['root']))