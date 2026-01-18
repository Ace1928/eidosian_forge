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
def _permutation_distribution_t(data, permutations, size_a, equal_var, random_state=None):
    """Generation permutation distribution of t statistic"""
    random_state = check_random_state(random_state)
    size = data.shape[-1]
    n_max = special.comb(size, size_a)
    if permutations < n_max:
        perm_generator = (random_state.permutation(size) for i in range(permutations))
    else:
        permutations = n_max
        perm_generator = (np.concatenate(z) for z in _all_partitions(size_a, size - size_a))
    t_stat = []
    for indices in _batch_generator(perm_generator, batch=50):
        indices = np.array(indices)
        data_perm = data[..., indices]
        data_perm = np.moveaxis(data_perm, -2, 0)
        a = data_perm[..., :size_a]
        b = data_perm[..., size_a:]
        t_stat.append(_calc_t_stat(a, b, equal_var))
    t_stat = np.concatenate(t_stat, axis=0)
    return (t_stat, permutations, n_max)