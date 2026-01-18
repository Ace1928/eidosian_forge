from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class _MapAndBatchDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that maps a function over a batch of elements."""

    def __init__(self, input_dataset, map_func, batch_size, num_parallel_calls, drop_remainder, use_legacy_function=False):
        self._input_dataset = input_dataset
        self._map_func = structured_function.StructuredFunctionWrapper(map_func, 'tf.data.experimental.map_and_batch()', dataset=input_dataset, use_legacy_function=use_legacy_function)
        self._batch_size_t = ops.convert_to_tensor(batch_size, dtype=dtypes.int64, name='batch_size')
        self._num_parallel_calls_t = ops.convert_to_tensor(num_parallel_calls, dtype=dtypes.int64, name='num_parallel_calls')
        self._drop_remainder_t = ops.convert_to_tensor(drop_remainder, dtype=dtypes.bool, name='drop_remainder')
        constant_drop_remainder = tensor_util.constant_value(self._drop_remainder_t)
        if constant_drop_remainder:
            self._element_spec = nest.map_structure(lambda component_spec: component_spec._batch(tensor_util.constant_value(self._batch_size_t)), self._map_func.output_structure)
        else:
            self._element_spec = nest.map_structure(lambda component_spec: component_spec._batch(None), self._map_func.output_structure)
        variant_tensor = ged_ops.map_and_batch_dataset(self._input_dataset._variant_tensor, self._map_func.function.captured_inputs, f=self._map_func.function, batch_size=self._batch_size_t, num_parallel_calls=self._num_parallel_calls_t, drop_remainder=self._drop_remainder_t, preserve_cardinality=True, **self._flat_structure)
        super(_MapAndBatchDataset, self).__init__(input_dataset, variant_tensor)

    def _functions(self):
        return [self._map_func]

    @property
    def element_spec(self):
        return self._element_spec