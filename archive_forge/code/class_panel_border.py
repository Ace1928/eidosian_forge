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
class panel_border(MixinSequenceOfValues):
    """
    Panel border

    Parameters
    ----------
    theme_element : element_rect
    """
    _omit = ['facecolor']

    def apply_figure(self, figure: Figure, targets: ThemeTargets):
        super().apply_figure(figure, targets)
        if not (rects := targets.panel_border):
            return
        d = self.properties
        with suppress(KeyError):
            if d['edgecolor'] == 'none' or d['size'] == 0:
                return
        if 'edgecolor' in d and 'alpha' in d:
            d['edgecolor'] = to_rgba(d['edgecolor'], d['alpha'])
            del d['alpha']
        self.set(rects, d)

    def blank_figure(self, figure: Figure, targets: ThemeTargets):
        super().blank_figure(figure, targets)
        for rect in targets.panel_border:
            rect.set_visible(False)