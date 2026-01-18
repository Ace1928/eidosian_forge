from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class GridNodeHighlightConfig:
    """
    Configurations for Persistent Grid Highlight
    """
    grid_highlight_config: GridHighlightConfig
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['gridHighlightConfig'] = self.grid_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(grid_highlight_config=GridHighlightConfig.from_json(json['gridHighlightConfig']), node_id=dom.NodeId.from_json(json['nodeId']))