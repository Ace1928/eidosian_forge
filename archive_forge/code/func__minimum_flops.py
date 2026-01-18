import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Minimum', 'flops')
def _minimum_flops(graph, node):
    """Compute flops for Minimum operation."""
    return _binary_per_element_op_flops(graph, node)