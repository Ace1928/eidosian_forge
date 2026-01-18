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
class text(axis_text, legend_text, strip_text, title):
    """
    All text elements in the plot

    Parameters
    ----------
    theme_element : element_text
    """

    @property
    def rcParams(self) -> dict[str, Any]:
        rcParams = super().rcParams
        family = self.properties.get('family')
        style = self.properties.get('style')
        weight = self.properties.get('weight')
        size = self.properties.get('size')
        color = self.properties.get('color')
        if family:
            rcParams['font.family'] = family
        if style:
            rcParams['font.style'] = style
        if weight:
            rcParams['font.weight'] = weight
        if size:
            rcParams['font.size'] = size
            rcParams['xtick.labelsize'] = size
            rcParams['ytick.labelsize'] = size
            rcParams['legend.fontsize'] = size
        if color:
            rcParams['text.color'] = color
        return rcParams