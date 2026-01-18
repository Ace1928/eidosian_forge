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
class axis_text_y(MixinSequenceOfValues):
    """
    y-axis tick labels

    Parameters
    ----------
    theme_element : element_text
    """
    _omit = ['margin']

    def apply_ax(self, ax: Axes):
        super().apply_ax(ax)
        self.set(ax.get_yticklabels())

    def blank_ax(self, ax: Axes):
        super().blank_ax(ax)
        ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)