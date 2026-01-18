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
def _lift_single_variable(old_variable, graph, variable_holder):
    """Lifts `old_variable` out of the `FuncGraph` `graph`."""
    new_variable = resource_variable_ops.UninitializedVariable(shape=old_variable.shape, dtype=old_variable.dtype, name=old_variable.op.name, trainable=old_variable.trainable, extra_handle_data=old_variable.handle)
    new_variable._initializer_op = old_variable._initializer_op
    graph.add_capture(new_variable.handle, old_variable.handle)
    graph.capture(new_variable.handle)
    variable_name = new_variable.name.split(':')[0]
    variable_holder._variables_by_name[variable_name] = new_variable
    graph._weak_variables.append(weakref.ref(new_variable))
    graph.watch_variable(new_variable)
    return new_variable