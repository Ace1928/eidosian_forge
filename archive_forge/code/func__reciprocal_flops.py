import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Reciprocal', 'flops')
def _reciprocal_flops(graph, node):
    """Compute flops for Reciprocal operation."""
    return _unary_op_flops(graph, node)