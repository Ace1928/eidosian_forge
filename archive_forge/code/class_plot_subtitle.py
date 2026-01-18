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
class plot_subtitle(themeable):
    """
    Plot subtitle

    Parameters
    ----------
    theme_element : element_text

    Notes
    -----
    The default horizontal alignment for the subtitle is left. And when
    it is present, by default it drags the title to the left. The subtitle
    drags the title to the left only if none of the two has their horizontal
    alignment are set.
    """
    _omit = ['margin']

    def apply_figure(self, figure: Figure, targets: ThemeTargets):
        super().apply_figure(figure, targets)
        if (text := targets.plot_subtitle):
            text.set(**self.properties)

    def blank_figure(self, figure: Figure, targets: ThemeTargets):
        super().blank_figure(figure, targets)
        if (text := targets.plot_subtitle):
            text.set_visible(False)