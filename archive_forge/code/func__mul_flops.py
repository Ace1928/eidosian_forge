import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Mul', 'flops')
def _mul_flops(graph, node):
    """Compute flops for Mul operation."""
    return _binary_per_element_op_flops(graph, node)