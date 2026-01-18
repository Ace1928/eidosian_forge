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
class title(axis_title, legend_title, plot_title, plot_subtitle, plot_caption):
    """
    All titles on the plot

    Parameters
    ----------
    theme_element : element_text
    """