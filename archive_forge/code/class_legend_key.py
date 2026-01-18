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
class legend_key(MixinSequenceOfValues):
    """
    Legend key background

    Parameters
    ----------
    theme_element : element_rect
    """

    def apply_figure(self, figure: Figure, targets: ThemeTargets):
        super().apply_figure(figure, targets)
        properties = self.properties
        if properties.get('edgecolor') in ('none', 'None'):
            properties['linewidth'] = 0
        rects = [da.patch for da in targets.legend_key]
        self.set(rects, properties)

    def blank_figure(self, figure: Figure, targets: ThemeTargets):
        super().blank_figure(figure, targets)
        for da in targets.legend_key:
            da.patch.set_visible(False)