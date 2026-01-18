import warnings
from numbers import Integral, Real
import numpy as np
from scipy.sparse import SparseEfficiencyWarning, issparse
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..exceptions import DataConversionWarning
from ..metrics import pairwise_distances
from ..metrics.pairwise import _VALID_METRICS, PAIRWISE_BOOLEAN_FUNCTIONS
from ..neighbors import NearestNeighbors
from ..utils import gen_batches, get_chunk_n_rows
from ..utils._param_validation import (
from ..utils.validation import check_memory
def _correct_predecessor(reachability_plot, predecessor_plot, ordering, s, e):
    """Correct for predecessors.

    Applies Algorithm 2 of [1]_.

    Input parameters are ordered by the computer OPTICS ordering.

    .. [1] Schubert, Erich, Michael Gertz.
       "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
       the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.
    """
    while s < e:
        if reachability_plot[s] > reachability_plot[e]:
            return (s, e)
        p_e = predecessor_plot[e]
        for i in range(s, e):
            if p_e == ordering[i]:
                return (s, e)
        e -= 1
    return (None, None)