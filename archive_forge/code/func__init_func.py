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
@def_function.function()
def _init_func():
    """Creates an iterator for the input dataset.

      Returns:
        A `string` tensor that encapsulates the iterator created.
      """
    ds_variant = gen_dataset_ops.unwrap_dataset_variant(wrap_ds_variant)
    resource = gen_dataset_ops.anonymous_iterator(**self._input_dataset._flat_structure)
    with ops.control_dependencies([gen_dataset_ops.make_iterator(ds_variant, resource)]):
        return gen_dataset_ops.iterator_to_string_handle(resource)