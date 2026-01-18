import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('BiasAddGrad', 'flops')
def _bias_add_grad_flops(graph, node):
    """Compute flops for BiasAddGrad operation."""
    return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)