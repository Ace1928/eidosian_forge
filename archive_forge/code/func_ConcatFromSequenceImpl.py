from __future__ import annotations
import typing
import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.model import expect
def ConcatFromSequenceImpl(sequence: list[np.ndarray], axis: int, new_axis: int | None=0) -> np.ndarray:
    if not new_axis:
        return np.concatenate(sequence, axis)
    return np.stack(sequence, axis)