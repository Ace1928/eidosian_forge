import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Pow', 'flops')
def _pow_flops(graph, node):
    """Compute flops for Pow operation."""
    return _binary_per_element_op_flops(graph, node)