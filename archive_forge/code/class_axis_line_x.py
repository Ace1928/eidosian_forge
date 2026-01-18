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
class axis_line_x(themeable):
    """
    x-axis line

    Parameters
    ----------
    theme_element : element_line
    """
    position = 'bottom'
    _omit = ['solid_capstyle']

    def apply_ax(self, ax: Axes):
        super().apply_ax(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set(**self.properties)

    def blank_ax(self, ax: Axes):
        super().blank_ax(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)