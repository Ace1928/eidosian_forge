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
class plot_margin_right(themeable):
    """
    Plot Margin on the right

    Parameters
    ----------
    theme_element : float
        Must be in the [0, 1] range. It is specified
        as a fraction of the figure width and figure
        height.
    """