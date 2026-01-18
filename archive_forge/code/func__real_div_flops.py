import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('RealDiv', 'flops')
def _real_div_flops(graph, node):
    """Compute flops for RealDiv operation."""
    return _binary_per_element_op_flops(graph, node)