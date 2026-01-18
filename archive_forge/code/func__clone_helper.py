from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.data.util import traverse
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.types import data as data_types
from tensorflow.python.types import distribute as distribute_types
def _clone_helper(op_to_clone, variant_tensor_ops):
    """Helper method that recursively clones `op_to_clone`.

  Args:
    op_to_clone: The op we want to clone.
    variant_tensor_ops: A list of ops that we have to clone along the way.

  Returns:
    A dictionary mapping old_ops to new_ops created. Includes op_to_clone
    as a key.
  """
    remap_dict = {}
    for input_tensor in op_to_clone.inputs:
        input_tensor_op = input_tensor.op
        if input_tensor_op in variant_tensor_ops:
            recursive_map = _clone_helper(input_tensor_op, variant_tensor_ops)
            remap_dict.update(recursive_map)
    inputs_list = []
    for input_tensor in op_to_clone.inputs:
        input_tensor_op = input_tensor.op
        if input_tensor_op in remap_dict:
            remapped_input = remap_dict[input_tensor_op].outputs[0]
            inputs_list.append(remapped_input)
        else:
            inputs_list.append(input_tensor_op.outputs[input_tensor.value_index])
    g = ops.get_default_graph()
    new_op = g.create_op(op_to_clone.type, inputs_list, [o.dtype for o in op_to_clone.outputs], name=op_to_clone.name, attrs=op_to_clone.node_def.attr, op_def=_get_op_def(op_to_clone))
    remap_dict[op_to_clone] = new_op
    return remap_dict