import warnings
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
def _map_v2(input_dataset, map_func, num_parallel_calls=None, deterministic=None, name=None):
    """See `Dataset.map()` for details."""
    if num_parallel_calls is None or debug_mode.DEBUG_MODE:
        if deterministic is not None and (not debug_mode.DEBUG_MODE):
            warnings.warn('The `deterministic` argument has no effect unless the `num_parallel_calls` argument is specified.')
        return _MapDataset(input_dataset, map_func, preserve_cardinality=True, name=name)
    else:
        return _ParallelMapDataset(input_dataset, map_func, num_parallel_calls=num_parallel_calls, deterministic=deterministic, preserve_cardinality=True, name=name)