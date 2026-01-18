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
class axis_ticks_minor_y(MixinSequenceOfValues):
    """
    y-axis minor tick lines

    Parameters
    ----------
    theme_element : element_line
    """

    def apply_ax(self, ax: Axes):
        super().apply_ax(ax)
        params = ax.yaxis.get_tick_params(which='minor')
        if not params.get('left', False):
            return
        tick_params = {}
        properties = self.properties
        with suppress(KeyError):
            tick_params['width'] = properties.pop('linewidth')
        with suppress(KeyError):
            tick_params['color'] = properties.pop('color')
        if tick_params:
            ax.yaxis.set_tick_params(which='minor', **tick_params)
        lines = [t.tick1line for t in ax.yaxis.get_minor_ticks()]
        self.set(lines, properties)

    def blank_ax(self, ax: Axes):
        super().blank_ax(ax)
        for tick in ax.yaxis.get_minor_ticks():
            tick.tick1line.set_visible(False)