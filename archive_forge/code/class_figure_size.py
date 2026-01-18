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
class figure_size(themeable):
    """
    Figure size in inches

    Parameters
    ----------
    theme_element : tuple
        (width, height) in inches
    """

    def apply_figure(self, figure: Figure, targets: ThemeTargets):
        figure.set_size_inches(self.properties['value'])