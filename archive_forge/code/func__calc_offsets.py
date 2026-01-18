import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
@staticmethod
def _calc_offsets(sizes, k):
    return np.cumsum([0, *sizes @ [k, 1]])