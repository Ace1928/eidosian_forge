from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops
class _AssertNextDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that asserts which transformations happen next."""

    def __init__(self, input_dataset, transformations):
        """See `assert_next()` for details."""
        self._input_dataset = input_dataset
        if transformations is None:
            raise ValueError('Invalid `transformations`. `transformations` should not be empty.')
        self._transformations = ops.convert_to_tensor(transformations, dtype=dtypes.string, name='transformations')
        variant_tensor = gen_experimental_dataset_ops.experimental_assert_next_dataset(self._input_dataset._variant_tensor, self._transformations, **self._flat_structure)
        super(_AssertNextDataset, self).__init__(input_dataset, variant_tensor)