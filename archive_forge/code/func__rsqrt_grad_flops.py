import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('RsqrtGrad', 'flops')
def _rsqrt_grad_flops(graph, node):
    """Compute flops for RsqrtGrad operation."""
    return _binary_per_element_op_flops(graph, node, ops_per_element=4)