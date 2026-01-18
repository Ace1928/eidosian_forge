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
class legend_justification_top(themeable):
    """
    Justification of legends placed at the top

    Parameters
    ----------
    theme_element : Literal["left", "center", "right"] | float
        How to justify the entire group with 1 or more guides. i.e. How
        to slide the legend along the top row.
        If a float, it should be in the range `[0, 1]`, where
        `0` is `"left"` and `1` is `"right"`.
    """