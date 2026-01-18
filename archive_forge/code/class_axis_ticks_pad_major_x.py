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
class axis_ticks_pad_major_x(themeable):
    """
    x-axis major-tick padding

    Parameters
    ----------
    theme_element : float
        Value in points.
    """

    def apply_ax(self, ax: Axes):
        super().apply_ax(ax)
        val = self.properties['value']
        for t in ax.xaxis.get_major_ticks():
            _val = val if t.tick1line.get_visible() else 0
            t.set_pad(_val)