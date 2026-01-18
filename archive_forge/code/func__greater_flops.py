import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Greater', 'flops')
def _greater_flops(graph, node):
    """Compute flops for Greater operation."""
    return _binary_per_element_op_flops(graph, node)