import weakref
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _filter_returned_ops(fn):
    """Filtering out any ops returned by function.

  Args:
    fn: a function

  Returns:
    A tuple of (
      Wrapped function that returns `None` in place of any ops,
      dict that maps the index in the flat output structure to the returned op
    )
  """
    returned_ops = {}

    def wrap_and_filter_returned_ops(*args, **kwargs):
        outputs = fn(*args, **kwargs)
        flat_outputs = nest.flatten(outputs)
        for n in range(len(flat_outputs)):
            output = flat_outputs[n]
            if isinstance(output, ops.Operation):
                returned_ops[n] = output
                flat_outputs[n] = None
        return nest.pack_sequence_as(outputs, flat_outputs)
    return (wrap_and_filter_returned_ops, returned_ops)