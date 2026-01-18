from __future__ import annotations
from enum import IntEnum
from typing import Callable
import numpy as np
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
class PostTransform(IntEnum):
    NONE = 0
    SOFTMAX = 1
    LOGISTIC = 2
    SOFTMAX_ZERO = 3
    PROBIT = 4