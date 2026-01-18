import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('LessEqual', 'flops')
def _less_equal_flops(graph, node):
    """Compute flops for LessEqual operation."""
    return _binary_per_element_op_flops(graph, node)