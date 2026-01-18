import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
class TensorHandle:

    def __init__(self, data, dtype):
        self.data = data
        self.dtype = dtype

    def __bool__(self):
        return bool(self.data.all())