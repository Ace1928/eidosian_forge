import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
def _gen_gaussians(center_locs, sigmas, total_length):
    xdata = np.arange(0, total_length).astype(float)
    out_data = np.zeros(total_length, dtype=float)
    for ind, sigma in enumerate(sigmas):
        tmp = (xdata - center_locs[ind]) / sigma
        out_data += np.exp(-tmp ** 2)
    return out_data