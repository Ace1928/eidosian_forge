from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops
class _FilterDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that filters its input according to a predicate function."""

    def __init__(self, input_dataset, predicate, use_legacy_function=False, name=None):
        """See `Dataset.filter` for details."""
        self._input_dataset = input_dataset
        wrapped_func = structured_function.StructuredFunctionWrapper(predicate, self._transformation_name(), dataset=input_dataset, use_legacy_function=use_legacy_function)
        if not wrapped_func.output_structure.is_compatible_with(tensor_spec.TensorSpec([], dtypes.bool)):
            raise ValueError(f'Invalid `predicate`. `predicate` must return a `tf.bool` scalar tensor, but its return type is {wrapped_func.output_structure}.')
        self._predicate = wrapped_func
        self._name = name
        variant_tensor = gen_dataset_ops.filter_dataset(input_dataset._variant_tensor, other_arguments=self._predicate.function.captured_inputs, predicate=self._predicate.function, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    def _functions(self):
        return [self._predicate]

    def _transformation_name(self):
        return 'Dataset.filter()'