from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
def _linear_coeffs(ratio: float, scale: float | None=None) -> np.ndarray:
    del scale
    return np.array([1 - ratio, ratio])