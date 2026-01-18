import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Less', 'flops')
def _less_flops(graph, node):
    """Compute flops for Less operation."""
    return _binary_per_element_op_flops(graph, node)