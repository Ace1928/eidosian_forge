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
class legend_ticks_length(themeable):
    """
    Length of ticks in the legend

    Parameters
    ----------
    theme_element : float
        A good value should be in the range `[0, 0.5]`.
    """