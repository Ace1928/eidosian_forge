import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Neg', 'flops')
def _neg_flops(graph, node):
    """Compute flops for Neg operation."""
    return _unary_op_flops(graph, node)