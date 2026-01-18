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
def _check_num_inputs_outputs(cond_graph, body_graph, num_flattened_loop_vars):
    """Checks the number of inputs/outputs of `cond_graph` and `body_graph`."""
    assert len(cond_graph.inputs) == num_flattened_loop_vars, 'cond_graph takes %d inputs; Expected: %d' % (len(cond_graph.inputs), num_flattened_loop_vars)
    assert len(cond_graph.outputs) == 1, 'cond_graph has %d outputs; Expected: 1' % len(cond_graph.outputs)
    assert len(body_graph.inputs) == num_flattened_loop_vars, 'body_graph takes %d inputs; Expected: %d' % (len(body_graph.inputs), num_flattened_loop_vars)
    assert len(body_graph.outputs) == num_flattened_loop_vars, 'body_graph has %d outputs; Expected: %d' % (len(body_graph.outputs), num_flattened_loop_vars)