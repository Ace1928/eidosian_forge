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
def _log_bins(xmin, xmax, bins):
    if xmin > xmax:
        return np.array([xmax])
    xmin = np.log10(xmin)
    xmax = np.log10(xmax)
    xrange = max(xmax - xmin, 1)
    xmin = max(xmin - xrange * 0.1, np.log10(_EPS))
    xmax = xmax + xrange * 0.1
    return np.logspace(xmin, xmax, bins + 1)