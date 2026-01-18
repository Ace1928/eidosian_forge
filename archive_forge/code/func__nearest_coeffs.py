from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
def _nearest_coeffs(ratio: float | int | np.ndarray, mode: str='round_prefer_floor') -> np.ndarray:
    if isinstance(ratio, int) or ratio.is_integer():
        return np.array([0, 1])
    if mode == 'round_prefer_floor':
        return np.array([ratio <= 0.5, ratio > 0.5])
    if mode == 'round_prefer_ceil':
        return np.array([ratio < 0.5, ratio >= 0.5])
    if mode == 'floor':
        return np.array([1, 0])
    if mode == 'ceil':
        return np.array([0, 1])
    raise ValueError(f'Unexpected value {mode!r}.')