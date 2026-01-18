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
def _next_func(string_handle):
    """Calls get_next for created iterator.

      Args:
        string_handle: An iterator string handle created by _init_func
      Returns:
        The elements generated from `input_dataset`
      """
    with ops.device(self._source_device_string):
        iterator = iterator_ops.Iterator.from_string_handle(string_handle, dataset_ops.get_legacy_output_types(self), dataset_ops.get_legacy_output_shapes(self), dataset_ops.get_legacy_output_classes(self))
    return structure.to_tensor_list(self.element_spec, iterator.get_next())