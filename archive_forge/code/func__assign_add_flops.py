import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('AssignAdd', 'flops')
def _assign_add_flops(graph, node):
    """Compute flops for AssignAdd operation."""
    return _unary_op_flops(graph, node)