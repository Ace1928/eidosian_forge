import numpy as np
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
@ops.RegisterStatistics('MaxPoolGrad', 'flops')
def _max_pool_grad_flops(graph, node):
    """Compute flops for MaxPoolGrad operation."""
    _verify_conv_data_format(node)
    kernel_shape = list(node.attr['ksize'].list.i)
    kernel_area = _list_product(kernel_shape)
    orig_out_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[1])
    orig_out_shape.assert_is_fully_defined()
    max_pool_ops = kernel_area * orig_out_shape.num_elements()
    return ops.OpStats('flops', max_pool_ops + orig_out_shape.num_elements())