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
class axis_ticks_pad_minor_x(themeable):
    """
    x-axis minor-tick padding

    Parameters
    ----------
    theme_element : float

    Note
    ----
    Padding is not applied when the
    [](`~plotnine.theme.themeables.axis_ticks_minor_x`) are
    blank, but it does apply when the
    [](`~plotnine.theme.themeables.axis_ticks_length_minor_x`) is zero.
    """

    def apply_ax(self, ax: Axes):
        super().apply_ax(ax)
        val = self.properties['value']
        for t in ax.xaxis.get_minor_ticks():
            _val = val if t.tick1line.get_visible() else 0
            t.set_pad(_val)