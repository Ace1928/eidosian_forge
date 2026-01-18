import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def _get_subplot_prop_name(subplot_type):
    """
    Get the name of the trace property used to associate a trace with a
    particular subplot type.  For most subplot types this is equal to the
    subplot type string. For example, the `scatter3d.scene` property is used
    to associate a `scatter3d` trace with a particular `scene` subplot.

    However, for some subplot types the trace property is not named after the
    subplot type.  For example, the `scatterpolar.subplot` property is used
    to associate a `scatterpolar` trace with a particular `polar` subplot.


    Parameters
    ----------
    subplot_type: str
        Subplot string value (e.g. 'scene4')

    Returns
    -------
    str
    """
    if subplot_type in _subplot_prop_named_subplot:
        subplot_prop_name = 'subplot'
    else:
        subplot_prop_name = subplot_type
    return subplot_prop_name