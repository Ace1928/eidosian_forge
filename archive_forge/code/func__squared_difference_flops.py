import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('SquaredDifference', 'flops')
def _squared_difference_flops(graph, node):
    """Compute flops for SquaredDifference operation."""
    return _binary_per_element_op_flops(graph, node, ops_per_element=2)