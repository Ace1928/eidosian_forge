import sys
from contextlib import suppress
import pandas as pd
from .._utils import array_kind
from ..exceptions import PlotnineError
from ..geoms import geom_blank
from ..mapping.aes import ALL_AESTHETICS, aes
from ..scales.scales import make_scale
class ylim(_lim):
    """
    Set y-axis limits

    Parameters
    ----------
    *limits :
        Min and max limits. Must be of size 2.
        You can also pass two values e.g
        `ylim(40, 100)`

    Notes
    -----
    If the 2nd value of `limits` is less than
    the first, a reversed scale will be created.
    """
    aesthetic = 'y'