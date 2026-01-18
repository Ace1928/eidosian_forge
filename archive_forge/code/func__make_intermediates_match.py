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
def _make_intermediates_match(branch_graphs, branch_optionals):
    """Returns new optionals lists that have matching signatures.

  This is done by mirroring each list in the other using none optionals.
  There is no merging of like optionals.

  Args:
    branch_graphs: `list` of `FuncGraph`.
    branch_optionals: `list` of `list`s of optional `Tensor`s from other
      branch_graphs

  Returns:
    A `list` of `list`s of `Tensor`s for each branch_graph. Each list has the
    same number of `Tensor`s, all of which will be optionals of the same
    shape/type.
  """
    new_branch_optionals = []
    intermediates_size = max((len(o) for o in branch_optionals))
    for i, branch_graph in enumerate(branch_graphs):
        other_optionals = _create_none_optionals(branch_graph, intermediates_size - len(branch_optionals[i]))
        new_branch_optionals.append(branch_optionals[i] + other_optionals)
    return new_branch_optionals