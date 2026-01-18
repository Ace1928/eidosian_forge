from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
class _PrefetchDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that asynchronously prefetches its input."""

    def __init__(self, input_dataset, buffer_size, slack_period=None, name=None):
        """See `Dataset.prefetch()` for details."""
        self._input_dataset = input_dataset
        if buffer_size is None:
            buffer_size = dataset_ops.AUTOTUNE
        self._buffer_size = ops.convert_to_tensor(buffer_size, dtype=dtypes.int64, name='buffer_size')
        self._name = name
        with ops.colocate_with(input_dataset._variant_tensor):
            variant_tensor = gen_dataset_ops.prefetch_dataset(input_dataset._variant_tensor, buffer_size=self._buffer_size, slack_period=slack_period, **self._common_args)
        super().__init__(input_dataset, variant_tensor)