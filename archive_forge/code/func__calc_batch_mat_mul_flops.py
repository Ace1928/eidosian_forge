import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('BatchMatMul', 'flops')
@ops.RegisterStatistics('BatchMatMulV2', 'flops')
@ops.RegisterStatistics('BatchMatMulV3', 'flops')
def _calc_batch_mat_mul_flops(graph, node):
    """Calculates the compute resources needed for BatchMatMul."""
    transpose_a = node.attr['transpose_a'].b
    a_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
    a_shape.assert_is_fully_defined()
    if transpose_a:
        k = int(a_shape[-2])
    else:
        k = int(a_shape[-1])
    output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
    output_shape.assert_is_fully_defined()
    output_count = np.prod(output_shape.as_list())
    return ops.OpStats('flops', k * output_count * 2)