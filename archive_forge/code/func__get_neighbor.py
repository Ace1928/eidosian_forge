from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
def _get_neighbor(x: float, n: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad `data` in 'edge' mode, and get n nearest elements in the padded array and their indexes in the original array.

    Args:
        x: Center index (in the unpadded coordinate system) of the found
            nearest elements.
        n: The number of neighbors.
        data: The array.

    Returns:
        A tuple containing the indexes of neighbor elements (the index
        can be smaller than 0 or higher than len(data)) and the value of
        these elements.
    """
    pad_width = np.ceil(n / 2).astype(int)
    padded = np.pad(data, pad_width, mode='edge')
    x += pad_width
    idxes = _get_neighbor_idxes(x, n, len(padded))
    ret = padded[idxes]
    return (idxes - pad_width, ret)