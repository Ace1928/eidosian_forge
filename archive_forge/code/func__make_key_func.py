from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _make_key_func(self, key_func, input_dataset):
    """Make wrapping defun for key_func."""
    self._key_func = structured_function.StructuredFunctionWrapper(key_func, self._transformation_name(), dataset=input_dataset)
    if not self._key_func.output_structure.is_compatible_with(tensor_spec.TensorSpec([], dtypes.int64)):
        raise ValueError(f'Invalid `key_func`. Expected `key_func` to return a scalar tf.int64 tensor, but instead `key_func` has output types={self._key_func.output_types} and shapes={self._key_func.output_shapes}.')