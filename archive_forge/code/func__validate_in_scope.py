import traceback
from typing import Any, Callable, Hashable
import weakref
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager.polymorphic_function import composite_tensor_utils
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _validate_in_scope(self, tensor):
    inner_graph = tensor.graph
    while inner_graph is not None and isinstance(inner_graph, FuncGraph):
        if inner_graph is self:
            try:
                tb = tensor.op.traceback
            except AttributeError:
                tensor_traceback = '<unknown>'
            else:
                tensor_traceback_list = []
                for frame in traceback.format_list(tb.get_user_frames()):
                    tensor_traceback_list.extend([f'  {line}' for line in frame.split('\n') if line.strip()])
                tensor_traceback = '\n'.join(tensor_traceback_list)
            raise errors.InaccessibleTensorError(f'{tensor!r} is out of scope and cannot be used here. Use return values, explicit Python locals or TensorFlow collections to access it.\nPlease see https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values for more information.\n\n{tensor!r} was defined here:\n{tensor_traceback}\n\nThe tensor {tensor!r} cannot be accessed from {self}, because it was defined in {tensor.graph}, which is out of scope.')
        inner_graph = inner_graph.outer_graph