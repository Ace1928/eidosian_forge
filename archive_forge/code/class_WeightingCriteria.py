from enum import IntEnum
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
class WeightingCriteria(IntEnum):
    NONE = 0
    TF = 1
    IDF = 2
    TFIDF = 3