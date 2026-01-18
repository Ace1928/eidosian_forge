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
class plot_caption(themeable):
    """
    Plot caption

    Parameters
    ----------
    theme_element : element_text
    """
    _omit = ['margin']

    def apply_figure(self, figure: Figure, targets: ThemeTargets):
        super().apply_figure(figure, targets)
        if (text := targets.plot_caption):
            text.set(**self.properties)

    def blank_figure(self, figure: Figure, targets: ThemeTargets):
        super().blank_figure(figure, targets)
        if (text := targets.plot_caption):
            text.set_visible(False)