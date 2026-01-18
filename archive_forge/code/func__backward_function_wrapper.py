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
def _backward_function_wrapper(*args):
    """Process output gradients and call the backward function."""
    if not backward.outputs:
        return backward.structured_outputs
    processed_args = []
    input_index = 0
    for output_index, arg in enumerate(args):
        if isinstance(arg, indexed_slices.IndexedSlices):
            arg = ops.convert_to_tensor(arg)
        if output_index in skip_positions:
            continue
        if arg is None:
            input_placeholder = backward.inputs[input_index]
            if input_placeholder.dtype == dtypes.variant:
                arg = variant_zeros_like[output_index]
            else:
                arg = array_ops.zeros(*default_gradient.shape_and_dtype(input_placeholder))
        processed_args.append(arg)
        input_index += 1
        if input_index >= backward_function_inputs:
            break
    return backward._call_flat(processed_args, remapped_captures)