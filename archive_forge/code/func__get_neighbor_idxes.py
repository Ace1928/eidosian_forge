from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
def _get_neighbor_idxes(x: float, n: int, limit: int) -> np.ndarray:
    """Return the n nearest indexes to x among `[0, limit)`,
    prefer the indexes smaller than x.
    As a result, the ratio must be in `(0, 1]`.

    Examples::

        get_neighbor_idxes(4, 2, 10) == [3, 4]
        get_neighbor_idxes(4, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.4, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.5, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.6, 3, 10) == [4, 5, 6]
        get_neighbor_idxes(4.4, 1, 10) == [4]
        get_neighbor_idxes(4.6, 1, 10) == [5]

    Args:
        x: float.
        n: the number of the wanted indexes.
        limit: the maximum value of index.

    Returns:
        An np.array containing n nearest indexes in ascending order
    """
    idxes = sorted(range(limit), key=lambda idx: (abs(x - idx), idx))[:n]
    idxes = sorted(idxes)
    return np.array(idxes)