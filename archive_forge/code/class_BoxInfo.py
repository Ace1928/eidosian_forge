import dataclasses
from typing import Optional, Tuple
import numpy as np
from onnx.reference.op_run import OpRun
class BoxInfo:

    def __init__(self, score: float=0, idx: int=-1):
        self.score_ = score
        self.idx_ = idx

    def __lt__(self, rhs) -> bool:
        return self.score_ < rhs.score_ or (self.score_ == rhs.score_ and self.idx_ > rhs.idx_)

    def __repr__(self) -> str:
        return f'BoxInfo({self.score_}, {self.idx_})'