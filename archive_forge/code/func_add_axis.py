import numpy as np
from ..processors import SequentialProcessor
def add_axis(x):
    return x[np.newaxis, ...]