import warnings
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
class _ParallelMapDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that maps a function over elements in its input in parallel."""

    def __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism=True, preserve_cardinality=False, use_legacy_function=False, name=None):
        """See `Dataset.map()` for details."""
        self._input_dataset = input_dataset
        self._use_inter_op_parallelism = use_inter_op_parallelism
        self._map_func = structured_function.StructuredFunctionWrapper(map_func, self._transformation_name(), dataset=input_dataset, use_legacy_function=use_legacy_function)
        if deterministic is None:
            self._deterministic = 'default'
        elif deterministic:
            self._deterministic = 'true'
        else:
            self._deterministic = 'false'
        self._preserve_cardinality = preserve_cardinality
        self._num_parallel_calls = ops.convert_to_tensor(num_parallel_calls, dtype=dtypes.int64, name='num_parallel_calls')
        self._name = name
        variant_tensor = gen_dataset_ops.parallel_map_dataset_v2(input_dataset._variant_tensor, self._map_func.function.captured_inputs, f=self._map_func.function, num_parallel_calls=self._num_parallel_calls, deterministic=self._deterministic, use_inter_op_parallelism=self._use_inter_op_parallelism, preserve_cardinality=self._preserve_cardinality, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    def _functions(self):
        return [self._map_func]

    @property
    def element_spec(self):
        return self._map_func.output_structure

    def _transformation_name(self):
        return 'Dataset.map()'