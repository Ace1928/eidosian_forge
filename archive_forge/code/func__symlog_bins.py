from .. import measure
from .. import utils
from .tools import label_axis
from .utils import _get_figure
from .utils import parse_fontsize
from .utils import show
from .utils import temp_fontsize
from scipy import sparse
import numbers
import numpy as np
def _symlog_bins(xmin, xmax, abs_min, bins):
    if xmin > 0:
        bins = _log_bins(xmin, xmax, bins)
    elif xmax < 0:
        bins = -1 * _log_bins(-xmax, -xmin, bins)[::-1]
    else:
        bins = max(bins, 3)
        if xmax > 0 and xmin < 0:
            bins = max(bins, 3)
            neg_range = np.log(-xmin) - np.log(abs_min)
            pos_range = np.log(xmax) - np.log(abs_min)
            total_range = pos_range + neg_range
            if total_range > 0:
                n_pos_bins = np.round((bins - 1) * pos_range / (pos_range + neg_range)).astype(int)
            else:
                n_pos_bins = 1
            n_neg_bins = max(bins - n_pos_bins - 1, 1)
        elif xmax > 0:
            bins = max(bins, 2)
            n_pos_bins = bins - 1
            n_neg_bins = 0
        elif xmin < 0:
            bins = max(bins, 2)
            n_neg_bins = bins - 1
            n_pos_bins = 0
        else:
            return np.array([-1, -0.1, 0.1, 1])
        pos_bins = _log_bins(abs_min, xmax, n_pos_bins)
        neg_bins = -1 * _log_bins(abs_min, -xmin, n_neg_bins)[::-1]
        bins = np.concatenate([neg_bins, pos_bins])
    return bins