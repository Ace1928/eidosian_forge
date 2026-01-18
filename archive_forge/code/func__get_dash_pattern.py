from __future__ import annotations
import itertools
import warnings
import numpy as np
from numpy.typing import ArrayLike
from pandas import Series
import matplotlib as mpl
from matplotlib.colors import to_rgb, to_rgba, to_rgba_array
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from seaborn._core.scales import Scale, Boolean, Continuous, Nominal, Temporal
from seaborn._core.rules import categorical_order, variable_type
from seaborn.palettes import QUAL_PALETTES, color_palette, blend_palette
from seaborn.utils import get_color_cycle
from typing import Any, Callable, Tuple, List, Union, Optional
@staticmethod
def _get_dash_pattern(style: str | DashPattern) -> DashPatternWithOffset:
    """Convert linestyle arguments to dash pattern with offset."""
    ls_mapper = {'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}
    if isinstance(style, str):
        style = ls_mapper.get(style, style)
        if style in ['solid', 'none', 'None']:
            offset = 0
            dashes = None
        elif style in ['dashed', 'dashdot', 'dotted']:
            offset = 0
            dashes = tuple(mpl.rcParams[f'lines.{style}_pattern'])
        else:
            options = [*ls_mapper.values(), *ls_mapper.keys()]
            msg = f'Linestyle string must be one of {options}, not {repr(style)}.'
            raise ValueError(msg)
    elif isinstance(style, tuple):
        if len(style) > 1 and isinstance(style[1], tuple):
            offset, dashes = style
        elif len(style) > 1 and style[1] is None:
            offset, dashes = style
        else:
            offset = 0
            dashes = style
    else:
        val_type = type(style).__name__
        msg = f'Linestyle must be str or tuple, not {val_type}.'
        raise TypeError(msg)
    if dashes is not None:
        try:
            dsum = sum(dashes)
        except TypeError as err:
            msg = f'Invalid dash pattern: {dashes}'
            raise TypeError(msg) from err
        if dsum:
            offset %= dsum
    return (offset, dashes)