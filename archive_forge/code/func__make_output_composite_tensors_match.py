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
def _make_output_composite_tensors_match(op_type, branch_graphs):
    """Modifies each branch_graph's outputs to have the same output signature.

  Currently the only transformation implemented is turning a Tensor into an
  equivalent IndexedSlices if the other branch returns an IndexedSlices.
  Updates branch_graph.{outputs,structured_outputs} for each branch_graph in
  branch_graphs.

  Args:
    op_type: _COND or _CASE
    branch_graphs: `list` of `FuncGraph`

  Raises:
    TypeError: if a set of outputs cannot be rewritten.
  """
    assert branch_graphs
    branch_outputs = [g.structured_outputs for g in branch_graphs]
    outputs_per_branch = list((len(outs) for outs in branch_outputs))
    assert len(set(outputs_per_branch)) == 1, outputs_per_branch
    for output_idx, branch_outs in enumerate(zip(*branch_outputs)):
        if len(set((type(out) for out in branch_outs))) == 1:
            continue
        if not any((isinstance(out, indexed_slices.IndexedSlices) for out in branch_outs)):
            continue
        for branch_idx, branch_out in enumerate(branch_outs):
            if isinstance(branch_out, indexed_slices.IndexedSlices):
                continue
            elif isinstance(branch_out, tensor_lib.Tensor):
                with branch_graphs[branch_idx].as_default():
                    branch_outputs[branch_idx][output_idx] = math_ops._as_indexed_slices(branch_out)
            else:
                raise TypeError('Cannot reconcile {op_name} {output_idx}-th outputs:\n  outputs from all branches: {outputs}'.format(op_name='tf.cond' if op_type == _COND else 'tf.switch_case', output_idx=output_idx, outputs=branch_outs))
    for branch_graph, branch_outs in zip(branch_graphs, branch_outputs):
        branch_graph.structured_outputs = branch_outs
        branch_graph.outputs = [t for t in func_graph_module.flatten(branch_outs) if t is not None]