from __future__ import annotations
from contextlib import suppress
from typing import TYPE_CHECKING
from warnings import warn
import numpy as np
from .._utils import to_rgba
from .._utils.registry import RegistryHierarchyMeta
from ..exceptions import PlotnineError, deprecated_themeable_name
from .elements import element_blank
from .elements.element_base import element_base
class axis_ticks_direction_x(themeable):
    """
    x-axis tick direction

    Parameters
    ----------
    theme_element : Literal["in", "out", "inout"]
        `in` for ticks inside the panel.
        `out` for ticks outside the panel.
        `inout` for ticks inside and outside the panel.
    """

    def apply_ax(self, ax: Axes):
        super().apply_ax(ax)
        ax.xaxis.set_tick_params(which='major', tickdir=self.properties['value'])