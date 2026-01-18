import sys
from contextlib import suppress
import pandas as pd
from .._utils import array_kind
from ..exceptions import PlotnineError
from ..geoms import geom_blank
from ..mapping.aes import ALL_AESTHETICS, aes
from ..scales.scales import make_scale
class shapelim(_lim):
    """
    Shapee limits
    """
    aesthetic = 'shape'