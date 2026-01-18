from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
def _cubic_coeffs(ratio: float, scale: float | None=None, A: float=-0.75) -> np.ndarray:
    del scale
    coeffs = [((A * (ratio + 1) - 5 * A) * (ratio + 1) + 8 * A) * (ratio + 1) - 4 * A, ((A + 2) * ratio - (A + 3)) * ratio * ratio + 1, ((A + 2) * (1 - ratio) - (A + 3)) * (1 - ratio) * (1 - ratio) + 1, ((A * (1 - ratio + 1) - 5 * A) * (1 - ratio + 1) + 8 * A) * (1 - ratio + 1) - 4 * A]
    return np.array(coeffs)