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
class bottom_spaces(_side_spaces):
    """
    Space in the figure for artists below the panels

    Ordered from the edge of the figure and going inwards
    """
    plot_margin: float = 0
    plot_caption: float = 0
    plot_caption_margin_top: float = 0
    legend: float = 0
    legend_box_spacing: float = 0
    axis_title_x: float = 0
    axis_title_x_margin_top: float = 0
    axis_xlabels: float = 0
    axis_xticks: float = 0

    def _calculate(self):
        pack = self.pack
        theme = self.pack.theme
        W, H = theme.getp('figure_size')
        F = W / H
        self.plot_margin = theme.getp('plot_margin_bottom') * F
        if pack.plot_caption:
            self.plot_caption = bbox_in_figure_space(pack.plot_caption, pack.figure, pack.renderer).height
            self.plot_caption_margin_top = theme.getp(('plot_caption', 'margin')).get_as('t', 'fig') * F
        if pack.legends and pack.legends.bottom:
            self.legend = self._legend_height
            self.legend_box_spacing = theme.getp('legend_box_spacing') * F
        if pack.axis_title_x:
            self.axis_title_x = bbox_in_figure_space(pack.axis_title_x, pack.figure, pack.renderer).height
            self.axis_title_x_margin_top = theme.getp(('axis_title_x', 'margin')).get_as('t', 'fig') * F
        self.axis_xticks = max_xticks_height(pack, 'last_row')
        self.axis_xlabels = max_xlabels_height(pack, 'last_row')
        protrusion = max_ylabels_bottom_protrusion(pack)
        adjustment = protrusion - (self.total - self.plot_margin)
        if adjustment > 0:
            self.plot_margin += adjustment

    @cached_property
    def _legend_size(self) -> TupleFloat2:
        if not (self.pack.legends and self.pack.legends.bottom):
            return (0, 0)
        bbox = bbox_in_figure_space(self.pack.legends.bottom.box, self.pack.figure, self.pack.renderer)
        return (bbox.width, bbox.height)

    def edge(self, item: str) -> float:
        """
        Distance w.r.t figure height from the bottom edge of the figure
        """
        return self.sum_upto(item)