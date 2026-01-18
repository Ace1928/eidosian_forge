from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.ops import gen_dataset_ops
def _flat_map(input_dataset, map_func, name=None):
    """See `Dataset.flat_map()` for details."""
    return _FlatMapDataset(input_dataset, map_func, name)