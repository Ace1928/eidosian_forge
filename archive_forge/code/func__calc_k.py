import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
@staticmethod
def _calc_k(sizes, total):
    rel_sum, abs_sum = sizes.sum(0)
    return (total - abs_sum) / rel_sum if rel_sum else 0