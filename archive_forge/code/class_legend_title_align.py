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
class legend_title_align(themeable):

    def __init__(self):
        msg = "Themeable 'legend_title_align' is deprecated. Use the horizontal and vertical alignment parameters ha & va of 'element_text' with 'lenged_title'."
        warn(msg, FutureWarning, stacklevel=1)