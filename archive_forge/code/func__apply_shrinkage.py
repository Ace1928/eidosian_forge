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
def _apply_shrinkage(self):
    """Multiply leaves values by shrinkage parameter.

        This must be done at the very end of the growing process. If this were
        done during the growing process e.g. in finalize_leaf(), then a leaf
        would be shrunk but its sibling would potentially not be (if it's a
        non-leaf), which would lead to a wrong computation of the 'middle'
        value needed to enforce the monotonic constraints.
        """
    for leaf in self.finalized_leaves:
        leaf.value *= self.shrinkage