from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_experimental_dataset_ops
class _UniqueDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A dataset containing the unique elements of an input dataset."""

    def __init__(self, input_dataset, name=None):
        """See `tf.data.Dataset.unique` for details."""
        self._input_dataset = input_dataset
        for ty in nest.flatten(dataset_ops.get_legacy_output_types(input_dataset)):
            if ty not in (dtypes.int32, dtypes.int64, dtypes.string):
                raise TypeError(f'`tf.data.Dataset.unique` does not support type {ty} -- only `tf.int32`, `tf.int64`, and `tf.string` are supported.')
        self._name = name
        variant_tensor = gen_experimental_dataset_ops.unique_dataset(self._input_dataset._variant_tensor, **self._common_args)
        super().__init__(input_dataset, variant_tensor)