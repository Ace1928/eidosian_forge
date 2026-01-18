import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Rsqrt', 'flops')
def _rsqrt_flops(graph, node):
    """Compute flops for Rsqrt operation."""
    return _unary_op_flops(graph, node, ops_per_element=2)