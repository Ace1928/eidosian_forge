import collections
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util_v1
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
def _build_while_op(loop_vars, cond_graph, body_graph, output_shapes, parallel_iterations, name, num_original_outputs):
    """Builds the functional StatelessWhile/While op."""
    cond_stateful_ops = [op for op in cond_graph.get_operations() if op._is_stateful]
    body_stateful_ops = [op for op in body_graph.get_operations() if op._is_stateful]
    if cond_stateful_ops or body_stateful_ops:
        op_fn = gen_functional_ops._while
    else:
        op_fn = gen_functional_ops.stateless_while

    def _make_op(inputs):
        while_op, tensors = util.get_op_and_outputs(op_fn(inputs, util.create_new_tf_function(cond_graph), util.create_new_tf_function(body_graph), output_shapes=output_shapes, parallel_iterations=parallel_iterations, name=name))
        _copy_handle_data(body_graph.outputs, tensors)
        util.maybe_set_lowering_attr(while_op)
        util.maybe_propagate_compile_time_consts_in_xla(while_op)
        _set_read_only_resource_inputs_attr(while_op, [cond_graph, body_graph])
        while_op._set_attr('_num_original_outputs', attr_value_pb2.AttrValue(i=num_original_outputs))
        cond_graph.outer_graph = ops.get_default_graph()
        body_graph.outer_graph = ops.get_default_graph()
        while_op._cond_graph = cond_graph
        while_op._body_graph = body_graph
        return tensors
    return util.run_as_function_for_tape_gradients(_make_op, loop_vars)