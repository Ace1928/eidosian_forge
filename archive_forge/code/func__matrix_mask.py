import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
def _matrix_mask(data, mask):
    """Ensure that data and mask are compatible and add missing values.

    Values will be plotted for cells where ``mask`` is ``False``.

    ``data`` is expected to be a DataFrame; ``mask`` can be an array or
    a DataFrame.

    """
    if mask is None:
        mask = np.zeros(data.shape, bool)
    if isinstance(mask, np.ndarray):
        if mask.shape != data.shape:
            raise ValueError('Mask must have the same shape as data.')
        mask = pd.DataFrame(mask, index=data.index, columns=data.columns, dtype=bool)
    elif isinstance(mask, pd.DataFrame):
        if not mask.index.equals(data.index) and mask.columns.equals(data.columns):
            err = 'Mask must have the same index and columns as data.'
            raise ValueError(err)
    mask = mask | pd.isnull(data)
    return mask