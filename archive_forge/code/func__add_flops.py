import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Add', 'flops')
@ops.RegisterStatistics('AddV2', 'flops')
def _add_flops(graph, node):
    """Compute flops for Add operation."""
    return _binary_per_element_op_flops(graph, node)