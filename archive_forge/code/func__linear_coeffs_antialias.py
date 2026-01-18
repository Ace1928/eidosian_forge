from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
def _linear_coeffs_antialias(ratio: float, scale: float) -> np.ndarray:
    scale = min(scale, 1.0)
    start = int(np.floor(-1 / scale) + 1)
    footprint = 2 - 2 * start
    args = (np.arange(start, start + footprint) - ratio) * scale
    coeffs = np.clip(1 - np.abs(args), 0, 1)
    return np.array(coeffs) / sum(coeffs)