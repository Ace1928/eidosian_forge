from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops
class _SleepDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that sleeps before producing each upstream element."""

    def __init__(self, input_dataset, sleep_microseconds):
        self._input_dataset = input_dataset
        self._sleep_microseconds = sleep_microseconds
        variant_tensor = gen_experimental_dataset_ops.sleep_dataset(self._input_dataset._variant_tensor, self._sleep_microseconds, **self._flat_structure)
        super(_SleepDataset, self).__init__(input_dataset, variant_tensor)