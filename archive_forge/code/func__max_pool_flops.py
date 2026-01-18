import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('MaxPool', 'flops')
def _max_pool_flops(graph, node):
    """Compute flops for MaxPool operation."""
    return _pool_flops(graph, node)