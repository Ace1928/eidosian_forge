from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops
class _NonSerializableDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that performs non-serializable identity transformation."""

    def __init__(self, input_dataset):
        """See `non_serializable()` for details."""
        self._input_dataset = input_dataset
        variant_tensor = gen_experimental_dataset_ops.experimental_non_serializable_dataset(self._input_dataset._variant_tensor, **self._flat_structure)
        super(_NonSerializableDataset, self).__init__(input_dataset, variant_tensor)