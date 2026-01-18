import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Softmax', 'flops')
def _softmax_flops(graph, node):
    """Compute flops for Softmax operation."""
    return _unary_op_flops(graph, node, ops_per_element=5)