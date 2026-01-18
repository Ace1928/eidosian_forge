from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class FlexItemHighlightConfig:
    """
    Configuration data for the highlighting of Flex item elements.
    """
    base_size_box: typing.Optional[BoxStyle] = None
    base_size_border: typing.Optional[LineStyle] = None
    flexibility_arrow: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.base_size_box is not None:
            json['baseSizeBox'] = self.base_size_box.to_json()
        if self.base_size_border is not None:
            json['baseSizeBorder'] = self.base_size_border.to_json()
        if self.flexibility_arrow is not None:
            json['flexibilityArrow'] = self.flexibility_arrow.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(base_size_box=BoxStyle.from_json(json['baseSizeBox']) if 'baseSizeBox' in json else None, base_size_border=LineStyle.from_json(json['baseSizeBorder']) if 'baseSizeBorder' in json else None, flexibility_arrow=LineStyle.from_json(json['flexibilityArrow']) if 'flexibilityArrow' in json else None)