import collections
import pprint
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def _rewrite_forward_and_call_backward(self, op, *doutputs):
    """Add outputs to the forward call and feed them to the grad function."""
    forward_function, backwards_function = self.forward_backward(len(doutputs))
    if not backwards_function.outputs:
        return backwards_function.structured_outputs
    op.graph._add_function_recursive(forward_function)
    op._set_func_attr('f', forward_function.name)
    op._set_type_list_attr('Tout', [o.dtype.as_datatype_enum for o in forward_function.function_type.flat_outputs])
    truncated_outputs = forward_function.function_type.flat_outputs[len(op.outputs):]
    op._add_outputs([o.dtype.as_datatype_enum for o in truncated_outputs], [o.shape for o in truncated_outputs])
    for i in range(len(op.outputs)):
        output_type = forward_function.function_type.flat_outputs[i]
        handle_data = output_type.dtype._handle_data
        if handle_data:
            handle_data_util.set_handle_data(op.outputs[i], handle_data.shape_inference)
    capture_mapping = dict(zip((ops.tensor_id(t) for t in self._func_graph.outputs), op.outputs))
    remapped_captures = [capture_mapping.get(ops.tensor_id(capture), capture) for capture in backwards_function.captured_inputs]
    cleaned_doutputs = []
    for doutput, placeholder in zip(doutputs, self._func_graph.outputs):
        if backprop_util.IsTrainable(placeholder):
            if isinstance(doutput, indexed_slices.IndexedSlices):
                cleaned_doutputs.append(ops.convert_to_tensor(doutput))
            elif doutput is not None:
                cleaned_doutputs.append(doutput)
            else:
                cleaned_doutputs.append(default_gradient.zeros_like(placeholder))
    return backwards_function._call_flat(cleaned_doutputs, remapped_captures)