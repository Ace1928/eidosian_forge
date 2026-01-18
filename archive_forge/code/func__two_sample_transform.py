import warnings
import math
from math import gcd
from collections import namedtuple
import numpy as np
from numpy import array, asarray, ma
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements
from scipy._lib._util import (check_random_state, MapWrapper, _get_nan,
import scipy.special as special
from scipy import linalg
from . import distributions
from . import _mstats_basic as mstats_basic
from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
from dataclasses import dataclass, field
from ._hypotests import _all_partitions
from ._stats_pythran import _compute_outer_prob_inside_method
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
from ._axis_nan_policy import (_axis_nan_policy_factory,
from ._binomtest import _binary_search_for_binom_tst as _binary_search
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats
from scipy.optimize import root_scalar
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy._lib._util import normalize_axis_index
from scipy._lib._util import float_factorial  # noqa: F401
from scipy.stats._mstats_basic import (  # noqa: F401
def _two_sample_transform(u, v):
    """Helper function that concatenates x and y for two sample MGC stat.

    See above for use.

    Parameters
    ----------
    u, v : ndarray
        `u` and `v` have shapes `(n, p)` and `(m, p)`.

    Returns
    -------
    x : ndarray
        Concatenate `u` and `v` along the `axis = 0`. `x` thus has shape
        `(2n, p)`.
    y : ndarray
        Label matrix for `x` where 0 refers to samples that comes from `u` and
        1 refers to samples that come from `v`. `y` thus has shape `(2n, 1)`.

    """
    nx = u.shape[0]
    ny = v.shape[0]
    x = np.concatenate([u, v], axis=0)
    y = np.concatenate([np.zeros(nx), np.ones(ny)], axis=0).reshape(-1, 1)
    return (x, y)