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
class legend_frame(themeable):
    """
    Frame around colorbar

    Parameters
    ----------
    theme_element : element_rect
    """
    _omit = ['facecolor']

    def apply_figure(self, figure: Figure, targets: ThemeTargets):
        super().apply_figure(figure, targets)
        if (rect := targets.legend_frame):
            rect.set(**self.properties)

    def blank_figure(self, figure: Figure, targets: ThemeTargets):
        super().blank_figure(figure, targets)
        if (rect := targets.legend_frame):
            rect.set_visible(False)