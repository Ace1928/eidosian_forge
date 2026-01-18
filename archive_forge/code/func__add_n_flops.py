import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('AddN', 'flops')
def _add_n_flops(graph, node):
    """Compute flops for AddN operation."""
    if not node.input:
        return _zero_flops(graph, node)
    in_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    in_shape.assert_is_fully_defined()
    return ops.OpStats('flops', in_shape.num_elements() * (len(node.input) - 1))