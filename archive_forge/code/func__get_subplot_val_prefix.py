import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def _get_subplot_val_prefix(subplot_type):
    """
    Get the subplot value prefix for a subplot type. For most subplot types
    this is equal to the subplot type string itself. For example, a
    `scatter3d.scene` value of `scene2` is used to associate the scatter3d
    trace with the `layout.scene2` subplot.

    However, the `xaxis`/`yaxis` subplot types are exceptions to this pattern.
    For example, a `scatter.xaxis` value of `x2` is used to associate the
    scatter trace with the `layout.xaxis2` subplot.

    Parameters
    ----------
    subplot_type: str
        Subplot string value (e.g. 'scene4')

    Returns
    -------
    str
    """
    if subplot_type == 'xaxis':
        subplot_val_prefix = 'x'
    elif subplot_type == 'yaxis':
        subplot_val_prefix = 'y'
    else:
        subplot_val_prefix = subplot_type
    return subplot_val_prefix