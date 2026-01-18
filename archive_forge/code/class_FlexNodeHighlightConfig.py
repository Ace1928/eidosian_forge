from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class FlexNodeHighlightConfig:
    flex_container_highlight_config: FlexContainerHighlightConfig
    node_id: dom.NodeId

    def to_json(self):
        json = dict()
        json['flexContainerHighlightConfig'] = self.flex_container_highlight_config.to_json()
        json['nodeId'] = self.node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(flex_container_highlight_config=FlexContainerHighlightConfig.from_json(json['flexContainerHighlightConfig']), node_id=dom.NodeId.from_json(json['nodeId']))