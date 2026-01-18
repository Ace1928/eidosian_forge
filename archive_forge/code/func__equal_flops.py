import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('Equal', 'flops')
def _equal_flops(graph, node):
    """Compute flops for Equal operation."""
    return _binary_per_element_op_flops(graph, node)