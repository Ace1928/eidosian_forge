from __future__ import annotations
from collections.abc import Sequence
import io
from typing import TYPE_CHECKING, Any, cast
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
from contourpy import FillType, LineType
from contourpy.convert import convert_filled, convert_lines
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.util.mpl_util import filled_to_mpl_paths, lines_to_mpl_paths
from contourpy.util.renderer import Renderer
def _autoscale(self) -> None:
    for ax in self._axes:
        if getattr(ax, '_need_autoscale', False):
            ax.autoscale_view(tight=True)
            ax._need_autoscale = False
    if self._want_tight and len(self._axes) > 1:
        self._fig.tight_layout()