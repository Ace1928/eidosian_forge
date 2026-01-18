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
class legend_background(themeable):
    """
    Legend background

    Parameters
    ----------
    theme_element : element_rect
    """

    def apply_figure(self, figure: Figure, targets: ThemeTargets):
        super().apply_figure(figure, targets)
        if (legends := targets.legends):
            properties = self.properties
            if properties.get('edgecolor') in ('none', 'None'):
                properties['linewidth'] = 0
            for aob in legends.boxes:
                aob.patch.set(**properties)
                if properties:
                    aob._drawFrame = True
                    if not aob.pad:
                        aob.pad = 0.2

    def blank_figure(self, figure: Figure, targets: ThemeTargets):
        super().blank_figure(figure, targets)
        if (legends := targets.legends):
            for aob in legends.boxes:
                aob.patch.set_visible(False)