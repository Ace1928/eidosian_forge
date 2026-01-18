from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class FlexContainerHighlightConfig:
    """
    Configuration data for the highlighting of Flex container elements.
    """
    container_border: typing.Optional[LineStyle] = None
    line_separator: typing.Optional[LineStyle] = None
    item_separator: typing.Optional[LineStyle] = None
    main_distributed_space: typing.Optional[BoxStyle] = None
    cross_distributed_space: typing.Optional[BoxStyle] = None
    row_gap_space: typing.Optional[BoxStyle] = None
    column_gap_space: typing.Optional[BoxStyle] = None
    cross_alignment: typing.Optional[LineStyle] = None

    def to_json(self):
        json = dict()
        if self.container_border is not None:
            json['containerBorder'] = self.container_border.to_json()
        if self.line_separator is not None:
            json['lineSeparator'] = self.line_separator.to_json()
        if self.item_separator is not None:
            json['itemSeparator'] = self.item_separator.to_json()
        if self.main_distributed_space is not None:
            json['mainDistributedSpace'] = self.main_distributed_space.to_json()
        if self.cross_distributed_space is not None:
            json['crossDistributedSpace'] = self.cross_distributed_space.to_json()
        if self.row_gap_space is not None:
            json['rowGapSpace'] = self.row_gap_space.to_json()
        if self.column_gap_space is not None:
            json['columnGapSpace'] = self.column_gap_space.to_json()
        if self.cross_alignment is not None:
            json['crossAlignment'] = self.cross_alignment.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(container_border=LineStyle.from_json(json['containerBorder']) if 'containerBorder' in json else None, line_separator=LineStyle.from_json(json['lineSeparator']) if 'lineSeparator' in json else None, item_separator=LineStyle.from_json(json['itemSeparator']) if 'itemSeparator' in json else None, main_distributed_space=BoxStyle.from_json(json['mainDistributedSpace']) if 'mainDistributedSpace' in json else None, cross_distributed_space=BoxStyle.from_json(json['crossDistributedSpace']) if 'crossDistributedSpace' in json else None, row_gap_space=BoxStyle.from_json(json['rowGapSpace']) if 'rowGapSpace' in json else None, column_gap_space=BoxStyle.from_json(json['columnGapSpace']) if 'columnGapSpace' in json else None, cross_alignment=LineStyle.from_json(json['crossAlignment']) if 'crossAlignment' in json else None)