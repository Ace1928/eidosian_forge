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
class legend_justification_inside(themeable):
    """
    Justification of legends placed inside the axes

    Parameters
    ----------
    theme_element : Literal["left", "right", "center", "top", "bottom"] |                     float | tuple[float, float]
        How to justify the entire group with 1 or more guides. i.e. What
        point of the legend box to place at the destination point in the
        panels area.

        If a float, it should be in the range `[0, 1]`, and it implies the
        horizontal part and with the vertical part fixed at `0.5`.

        Therefore a float value of `0.8` equivalent to a tuple value of
        `(0.8, 0.5)`.
    """