from typing import Tuple
import numpy as np
from onnx.reference.op_run import OpRun
class PreCalc:

    def __init__(self, pos1=0, pos2=0, pos3=0, pos4=0, w1=0, w2=0, w3=0, w4=0):
        self.pos1 = pos1
        self.pos2 = pos2
        self.pos3 = pos3
        self.pos4 = pos4
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def __repr__(self) -> str:
        return f'PreCalc({self.pos1},{self.pos2},{self.pos3},{self.pos4},{self.w1},{self.w2},{self.w3},{self.w4})'