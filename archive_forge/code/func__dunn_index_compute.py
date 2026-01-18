from itertools import combinations
from typing import Tuple
import torch
from torch import Tensor
def _dunn_index_compute(intercluster_distance: Tensor, max_intracluster_distance: Tensor) -> Tensor:
    """Compute the Dunn index based on updated state.

    Args:
        intercluster_distance: intercluster distances
        max_intracluster_distance: max intracluster distances

    Returns:
        scalar tensor with the dunn index

    """
    return intercluster_distance.min() / max_intracluster_distance.max()