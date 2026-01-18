import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
def _list_product(lst):
    """Computes product of element of the list."""
    result = 1
    for item in lst:
        result *= item
    return result