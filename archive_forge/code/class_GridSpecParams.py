from __future__ import annotations
import typing
from copy import deepcopy
from dataclasses import dataclass
from ._plot_side_space import LRTBSpaces, WHSpaceParts, calculate_panel_spacing
from .utils import bbox_in_figure_space, get_transPanels
@dataclass
class GridSpecParams:
    """
    Gridspec Parameters
    """
    left: float
    right: float
    top: float
    bottom: float
    wspace: float
    hspace: float