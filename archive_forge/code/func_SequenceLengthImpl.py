from __future__ import annotations
import typing
import numpy as np
import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.model import expect
def SequenceLengthImpl(sequence: list[np.ndarray]) -> np.int64:
    return np.int64(len(sequence))