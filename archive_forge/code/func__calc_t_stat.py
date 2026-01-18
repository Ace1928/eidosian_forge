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
def _calc_t_stat(a, b, equal_var, axis=-1):
    """Calculate the t statistic along the given dimension."""
    na = a.shape[axis]
    nb = b.shape[axis]
    avg_a = np.mean(a, axis=axis)
    avg_b = np.mean(b, axis=axis)
    var_a = _var(a, axis=axis, ddof=1)
    var_b = _var(b, axis=axis, ddof=1)
    if not equal_var:
        denom = _unequal_var_ttest_denom(var_a, na, var_b, nb)[1]
    else:
        denom = _equal_var_ttest_denom(var_a, na, var_b, nb)[1]
    return (avg_a - avg_b) / denom