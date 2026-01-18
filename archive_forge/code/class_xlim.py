import sys
from contextlib import suppress
import pandas as pd
from .._utils import array_kind
from ..exceptions import PlotnineError
from ..geoms import geom_blank
from ..mapping.aes import ALL_AESTHETICS, aes
from ..scales.scales import make_scale
class xlim(_lim):
    """
    Set x-axis limits

    Parameters
    ----------
    *limits :
        Min and max limits. Must be of size 2.
        You can also pass two values e.g
        `xlim(40, 100)`
    """
    aesthetic = 'x'