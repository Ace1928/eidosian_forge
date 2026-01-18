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
def _duplicate_body_captures_in_cond(cond_graph, body_graph_captures):
    """Creates placeholders for body captures in cond_graph.

  This is needed to match signatures of cond and body graphs.

  Args:
    cond_graph: cond branch graph
    body_graph_captures: Tensors which were captured when building the
      `body_graph`.
  """
    types = [t.dtype.as_datatype_enum for t in body_graph_captures]
    with cond_graph._c_graph.get() as c_graph:
        placeholders = c_api.TF_CreatePlaceholders(c_graph, types, compat.as_str(_build_cond_placeholders_name_prefix(cond_graph)))
    placeholder_ops = [ops.Operation._from_c_op(ph.oper, cond_graph) for ph in placeholders]
    tensors = []
    for op in placeholder_ops:
        tensors.append(op.outputs[0])
    tuples = zip(body_graph_captures, tensors)
    keys = [id(t) for t in body_graph_captures]
    for k, v in zip(keys, tuples):
        cond_graph._function_captures.add_or_replace(key=k, external=v[0], internal=v[1], is_by_ref=False)
    cond_graph.inputs.extend(tensors)