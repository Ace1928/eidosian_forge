import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('ArgMin', 'flops')
def _arg_min_flops(graph, node):
    """Compute flops for ArgMin operation."""
    return _reduction_op_flops(graph, node, reduce_flops=1, finalize_flops=0)