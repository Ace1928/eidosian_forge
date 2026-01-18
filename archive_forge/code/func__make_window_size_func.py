from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def _make_window_size_func(self, window_size_func):
    """Make wrapping defun for window_size_func."""

    def window_size_func_wrapper(key):
        return ops.convert_to_tensor(window_size_func(key), dtype=dtypes.int64)
    self._window_size_func = structured_function.StructuredFunctionWrapper(window_size_func_wrapper, self._transformation_name(), input_structure=tensor_spec.TensorSpec([], dtypes.int64))
    if not self._window_size_func.output_structure.is_compatible_with(tensor_spec.TensorSpec([], dtypes.int64)):
        raise ValueError(f'Invalid `window_size_func`. `window_size_func` must return a single `tf.int64` scalar tensor but its return type is {self._window_size_func.output_structure}.')