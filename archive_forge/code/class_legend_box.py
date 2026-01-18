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
class legend_box(themeable):
    """
    How to box up multiple legends

    Parameters
    ----------
    theme_element : Literal["vertical", "horizontal"]
        Whether to stack up the legends vertically or
        horizontally.
    """