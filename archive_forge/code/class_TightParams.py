from __future__ import annotations
import typing
from copy import deepcopy
from dataclasses import dataclass
from ._plot_side_space import LRTBSpaces, WHSpaceParts, calculate_panel_spacing
from .utils import bbox_in_figure_space, get_transPanels
@dataclass
class TightParams:
    """
    All parameters computed for the plotnine tight layout engine
    """
    sides: LRTBSpaces
    gullies: WHSpaceParts

    def __post_init__(self):
        self.params = GridSpecParams(left=self.sides.left, right=self.sides.right, top=self.sides.top, bottom=self.sides.bottom, wspace=self.gullies.wspace, hspace=self.gullies.hspace)

    def to_aspect_ratio(self, facet: facet, ratio: float, parts: WHSpaceParts) -> TightParams:
        """
        Modify TightParams to get a given aspect ratio
        """
        current_ratio = parts.h * parts.H / (parts.w * parts.W)
        increase_aspect_ratio = ratio > current_ratio
        if increase_aspect_ratio:
            return self._reduce_width(facet, ratio, parts)
        else:
            return self._reduce_height(facet, ratio, parts)

    def _reduce_height(self, facet: facet, ratio: float, parts: WHSpaceParts) -> TightParams:
        """
        Reduce the height of axes to get the aspect ratio
        """
        self = deepcopy(self)
        h1 = ratio * parts.w * (parts.W / parts.H)
        dh = (parts.h - h1) * facet.nrow / 2
        self.params.top -= dh
        self.params.bottom += dh
        self.params.hspace = parts.sh / h1
        self.sides.t.plot_margin += dh
        self.sides.b.plot_margin += dh
        return self

    def _reduce_width(self, facet: facet, ratio: float, parts: WHSpaceParts) -> TightParams:
        """
        Reduce the width of axes to get the aspect ratio
        """
        self = deepcopy(self)
        w1 = parts.h * parts.H / (ratio * parts.W)
        dw = (parts.w - w1) * facet.ncol / 2
        self.params.left += dw
        self.params.right -= dw
        self.params.wspace = parts.sw / w1
        self.sides.l.plot_margin += dw
        self.sides.r.plot_margin += dw
        return self