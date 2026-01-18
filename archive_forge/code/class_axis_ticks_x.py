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
class axis_ticks_x(axis_ticks_major_x, axis_ticks_minor_x):
    """
    x major and minor axis tick lines

    Parameters
    ----------
    theme_element : element_line
    """