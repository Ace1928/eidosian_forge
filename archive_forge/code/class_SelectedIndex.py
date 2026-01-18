import dataclasses
from typing import Optional, Tuple
import numpy as np
from onnx.reference.op_run import OpRun
class SelectedIndex:
    __slots__ = ('batch_index_', 'class_index_', 'box_index_')

    def __init__(self, batch_index: int=0, class_index: int=0, box_index: int=0) -> None:
        self.batch_index_ = batch_index
        self.class_index_ = class_index
        self.box_index_ = box_index