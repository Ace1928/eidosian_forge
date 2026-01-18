from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class LayoutShiftAttribution:
    previous_rect: dom.Rect
    current_rect: dom.Rect
    node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['previousRect'] = self.previous_rect.to_json()
        json['currentRect'] = self.current_rect.to_json()
        if self.node_id is not None:
            json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(previous_rect=dom.Rect.from_json(json['previousRect']), current_rect=dom.Rect.from_json(json['currentRect']), node_id=dom.BackendNodeId.from_json(json['nodeId']) if 'nodeId' in json else None)