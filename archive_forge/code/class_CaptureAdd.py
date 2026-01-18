from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class CaptureAdd(Capture):

    def __init__(self, left, right, ctx):
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self):
        return f'{self.left} + {self.right}'

    def execute(self):
        return get_val(self.left) + get_val(self.right)