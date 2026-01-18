import numpy as np
from collections import namedtuple
class GenericObject:

    def __init__(self, v):
        self.v = v

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self
    dtype = np.dtype('O')