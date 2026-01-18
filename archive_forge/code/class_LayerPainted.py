from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
@event_class('LayerTree.layerPainted')
@dataclass
class LayerPainted:
    layer_id: LayerId
    clip: dom.Rect

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LayerPainted:
        return cls(layer_id=LayerId.from_json(json['layerId']), clip=dom.Rect.from_json(json['clip']))