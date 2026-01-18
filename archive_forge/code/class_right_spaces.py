from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
@dataclass
class right_spaces(_side_spaces):
    """
    Space in the figure for artists on the right of the panels

    Ordered from the edge of the figure and going inwards
    """
    plot_margin: float = 0
    legend: float = 0
    legend_box_spacing: float = 0
    right_strip_width: float = 0

    def _calculate(self):
        pack = self.pack
        theme = self.pack.theme
        self.plot_margin = theme.getp('plot_margin_right')
        if pack.legends and pack.legends.right:
            self.legend = self._legend_width
            self.legend_box_spacing = theme.getp('legend_box_spacing')
        self.right_strip_width = get_right_strip_width(pack)
        protrusion = max_xlabels_right_protrusion(pack)
        adjustment = protrusion - (self.total - self.plot_margin)
        if adjustment > 0:
            self.plot_margin += adjustment

    @cached_property
    def _legend_size(self) -> TupleFloat2:
        if not (self.pack.legends and self.pack.legends.right):
            return (0, 0)
        bbox = bbox_in_figure_space(self.pack.legends.right.box, self.pack.figure, self.pack.renderer)
        return (bbox.width, bbox.height)

    def edge(self, item: str) -> float:
        """
        Distance w.r.t figure width from the right edge of the figure
        """
        return 1 - self.sum_upto(item)