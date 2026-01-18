from __future__ import annotations
import typing
from copy import deepcopy
from dataclasses import dataclass
from ._plot_side_space import LRTBSpaces, WHSpaceParts, calculate_panel_spacing
from .utils import bbox_in_figure_space, get_transPanels
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