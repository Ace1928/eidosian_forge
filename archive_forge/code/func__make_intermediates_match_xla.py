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
def _make_intermediates_match_xla(branch_graphs, branch_intermediates):
    """Like _make_intermediates_match but for the XLA case."""
    new_branch_intermediates = []
    for i, branch_graph in enumerate(branch_graphs):
        other_fakeparams = _create_fakeparams(branch_graph, sum((bi for bi in branch_intermediates if bi is not branch_intermediates[i]), []))
        num_preceding = sum((len(bi) for bi in branch_intermediates[:i]))
        new_branch_intermediates.append(other_fakeparams[:num_preceding] + branch_intermediates[i] + other_fakeparams[num_preceding:])
    return new_branch_intermediates