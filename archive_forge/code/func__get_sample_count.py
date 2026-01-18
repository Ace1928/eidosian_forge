import numbers
from itertools import chain
from math import ceil
import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles
from ...base import is_regressor
from ...utils import (
from ...utils._encode import _unique
from ...utils.parallel import Parallel, delayed
from .. import partial_dependence
from .._pd_utils import _check_feature_names, _get_feature_index
def _get_sample_count(self, n_samples):
    """Compute the number of samples as an integer."""
    if isinstance(self.subsample, numbers.Integral):
        if self.subsample < n_samples:
            return self.subsample
        return n_samples
    elif isinstance(self.subsample, numbers.Real):
        return ceil(n_samples * self.subsample)
    return n_samples