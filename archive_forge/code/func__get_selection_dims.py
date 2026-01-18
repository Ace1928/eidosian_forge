import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def _get_selection_dims(self):
    x0dim, y0dim, x1dim, y1dim = self.kdims
    invert_axes = self.opts.get('plot').kwargs.get('invert_axes', False)
    if invert_axes:
        x0dim, x1dim, y0dim, y1dim = (y0dim, y1dim, x0dim, x1dim)
    return (x0dim, y0dim, x1dim, y1dim)