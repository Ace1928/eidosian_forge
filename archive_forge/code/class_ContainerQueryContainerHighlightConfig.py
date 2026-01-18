from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class ContainerQueryContainerHighlightConfig:
    container_border: typing.Optional[LineStyle] = None
    descendant_border: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.container_border is not None:
            json['containerBorder'] = self.container_border.to_json()
        if self.descendant_border is not None:
            json['descendantBorder'] = self.descendant_border.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(container_border=LineStyle.from_json(json['containerBorder']) if 'containerBorder' in json else None, descendant_border=LineStyle.from_json(json['descendantBorder']) if 'descendantBorder' in json else None)