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
def _extend_region(steep_point, xward_point, start, min_samples):
    """Extend the area until it's maximal.

    It's the same function for both upward and downward reagions, depending on
    the given input parameters. Assuming:

        - steep_{upward/downward}: bool array indicating whether a point is a
          steep {upward/downward};
        - upward/downward: bool array indicating whether a point is
          upward/downward;

    To extend an upward reagion, ``steep_point=steep_upward`` and
    ``xward_point=downward`` are expected, and to extend a downward region,
    ``steep_point=steep_downward`` and ``xward_point=upward``.

    Parameters
    ----------
    steep_point : ndarray of shape (n_samples,), dtype=bool
        True if the point is steep downward (upward).

    xward_point : ndarray of shape (n_samples,), dtype=bool
        True if the point is an upward (respectively downward) point.

    start : int
        The start of the xward region.

    min_samples : int
       The same as the min_samples given to OPTICS. Up and down steep
       regions can't have more then ``min_samples`` consecutive non-steep
       points.

    Returns
    -------
    index : int
        The current index iterating over all the samples, i.e. where we are up
        to in our search.

    end : int
        The end of the region, which can be behind the index. The region
        includes the ``end`` index.
    """
    n_samples = len(steep_point)
    non_xward_points = 0
    index = start
    end = start
    while index < n_samples:
        if steep_point[index]:
            non_xward_points = 0
            end = index
        elif not xward_point[index]:
            non_xward_points += 1
            if non_xward_points > min_samples:
                break
        else:
            return end
        index += 1
    return end