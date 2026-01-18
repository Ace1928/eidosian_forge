import numbers
from heapq import heappop, heappush
from timeit import default_timer as time
import numpy as np
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from ._bitset import set_raw_bitset_from_binned_bitset
from .common import (
from .histogram import HistogramBuilder
from .predictor import TreePredictor
from .splitting import Splitter
from .utils import sum_parallel
def _compute_interactions(self, node):
    """Compute features allowed by interactions to be inherited by child nodes.

        Example: Assume constraints [{0, 1}, {1, 2}].
           1      <- Both constraint groups could be applied from now on
          / \\
         1   2    <- Left split still fulfills both constraint groups.
        / \\ / \\      Right split at feature 2 has only group {1, 2} from now on.

        LightGBM uses the same logic for overlapping groups. See
        https://github.com/microsoft/LightGBM/issues/4481 for details.

        Parameters:
        ----------
        node : TreeNode
            A node that might have children. Based on its feature_idx, the interaction
            constraints for possible child nodes are computed.

        Returns
        -------
        allowed_features : ndarray, dtype=uint32
            Indices of features allowed to split for children.
        interaction_cst_indices : list of ints
            Indices of the interaction sets that have to be applied on splits of
            child nodes. The fewer sets the stronger the constraint as fewer sets
            contain fewer features.
        """
    allowed_features = set()
    interaction_cst_indices = []
    for i in node.interaction_cst_indices:
        if node.split_info.feature_idx in self.interaction_cst[i]:
            interaction_cst_indices.append(i)
            allowed_features.update(self.interaction_cst[i])
    return (np.fromiter(allowed_features, dtype=np.uint32, count=len(allowed_features)), interaction_cst_indices)