from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as framework_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util.tf_export import tf_export
@def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
def _finalize_func(string_handle):
    """Destroys the iterator resource created.

      Args:
        string_handle: An iterator string handle created by _init_func
      Returns:
        Tensor constant 0
      """
    iterator_resource = gen_dataset_ops.iterator_from_string_handle_v2(string_handle, **self._input_dataset._flat_structure)
    with ops.control_dependencies([resource_variable_ops.destroy_resource_op(iterator_resource, ignore_lookup_error=True)]):
        return array_ops.constant(0, dtypes.int64)