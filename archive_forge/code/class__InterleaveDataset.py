import warnings
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
class _InterleaveDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that interleaves the result of transformed inputs."""

    def __init__(self, input_dataset, map_func, cycle_length, block_length, name=None):
        """See `Dataset.interleave()` for details."""
        self._input_dataset = input_dataset
        self._map_func = structured_function.StructuredFunctionWrapper(map_func, self._transformation_name(), dataset=input_dataset)
        if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):
            raise TypeError(f'The `map_func` argument must return a `Dataset` object. Got {dataset_ops.get_type(self._map_func.output_structure)!r}.')
        self._structure = self._map_func.output_structure._element_spec
        self._cycle_length = ops.convert_to_tensor(cycle_length, dtype=dtypes.int64, name='cycle_length')
        self._block_length = ops.convert_to_tensor(block_length, dtype=dtypes.int64, name='block_length')
        self._name = name
        variant_tensor = gen_dataset_ops.interleave_dataset(input_dataset._variant_tensor, self._map_func.function.captured_inputs, self._cycle_length, self._block_length, f=self._map_func.function, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    def _functions(self):
        return [self._map_func]

    @property
    def element_spec(self):
        return self._structure

    def _transformation_name(self):
        return 'Dataset.interleave()'