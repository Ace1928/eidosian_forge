import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('AssignSub', 'flops')
def _assign_sub_flops(graph, node):
    """Compute flops for AssignSub operation."""
    return _unary_op_flops(graph, node)