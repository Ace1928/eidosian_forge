import collections
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _resolve_grad_inputs(cond_graph, grad_graph):
    """Returns the tensors to pass as inputs to `grad_graph`.

  The `grad_graph` may have external references to
  1. Its outer graph containing the input gradients. These references are kept
     as is.
  2. Tensors in the forward pass graph. These tensors may not be "live"
     when the gradient is being computed. We replace such references by their
     corresponding tensor in `cond_graph.outer_graph`. In the case of nested
     control flow or functions, the gradient logic handling
     `grad_graph.outer_graph` will make sure the tensor from
     `cond_graph.outer_graph` is also correctly captured.

  Args:
    cond_graph: FuncGraph. The forward-pass function.
    grad_graph: FuncGraph. The gradients function.

  Returns:
    A list of inputs tensors to be passed to grad_graph.
  """
    new_inputs = []
    for t in grad_graph.external_captures:
        if t.graph != grad_graph.outer_graph:
            assert t.graph == cond_graph
            for i, output in enumerate(t.graph.outputs):
                if output is t:
                    t = t.graph._forward_cond.outputs[i]
                    break
            else:
                for i, output in enumerate(t.graph.internal_captures):
                    if output is t:
                        t = t.graph.external_captures[i]
                        break
                else:
                    raise ValueError('Could not find external tensor capture {tensor} in captures or outputs'.format(tensor=t))
            assert t.graph == cond_graph.outer_graph
        new_inputs.append(t)
    return new_inputs