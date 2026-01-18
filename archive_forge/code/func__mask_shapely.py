import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def _mask_shapely(masked_xvals, masked_yvals, geometry):
    from shapely.geometry import Point, Polygon
    points = (Point(x, y) for x, y in zip(masked_xvals, masked_yvals))
    poly = Polygon(geometry)
    return np.array([poly.contains(p) for p in points], dtype=bool)