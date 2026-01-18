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
class axis_ticks_pad_major(axis_ticks_pad_major_x, axis_ticks_pad_major_y):
    """
    Axis major-tick padding

    Parameters
    ----------
    theme_element : float
        Value in points.

    Note
    ----
    Padding is not applied when the
    [](`~plotnine.theme.themeables.axis_ticks_major`) are blank,
    but it does apply when the
    [](`~plotnine.theme.themeables.axis_ticks_length_major`) is zero.
    """