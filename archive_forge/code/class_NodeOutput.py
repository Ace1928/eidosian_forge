import logging
import json
import numpy as np
from mxnet import ndarray as nd
class NodeOutput:

    def __init__(self, name, dtype):
        self.name = name
        self.dtype = np.dtype(dtype)