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
class left_spaces(_side_spaces):
    """
    Space in the figure for artists on the left of the panels

    Ordered from the edge of the figure and going inwards
    """
    plot_margin: float = 0
    legend: float = 0
    legend_box_spacing: float = 0
    axis_title_y: float = 0
    axis_title_y_margin_right: float = 0
    axis_ylabels: float = 0
    axis_yticks: float = 0

    def _calculate(self):
        theme = self.pack.theme
        pack = self.pack
        self.plot_margin = theme.getp('plot_margin_left')
        if pack.legends and pack.legends.left:
            self.legend = self._legend_width
            self.legend_box_spacing = theme.getp('legend_box_spacing')
        if pack.axis_title_y:
            self.axis_title_y_margin_right = theme.getp(('axis_title_y', 'margin')).get_as('r', 'fig')
            self.axis_title_y = bbox_in_figure_space(pack.axis_title_y, pack.figure, pack.renderer).width
        self.axis_ylabels = max_ylabels_width(pack, 'first_col')
        self.axis_yticks = max_yticks_width(pack, 'first_col')
        protrusion = max_xlabels_left_protrusion(pack)
        adjustment = protrusion - (self.total - self.plot_margin)
        if adjustment > 0:
            self.plot_margin += adjustment

    @cached_property
    def _legend_size(self) -> TupleFloat2:
        if not (self.pack.legends and self.pack.legends.left):
            return (0, 0)
        bbox = bbox_in_figure_space(self.pack.legends.left.box, self.pack.figure, self.pack.renderer)
        return (bbox.width, bbox.height)

    def edge(self, item: str) -> float:
        """
        Distance w.r.t figure width from the left edge of the figure
        """
        return self.sum_upto(item)