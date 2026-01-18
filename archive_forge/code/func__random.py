import warnings
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def _random(seed=None, rerandomize_each_iteration=None, name=None):
    """See `Dataset.random()` for details."""
    return _RandomDataset(seed=seed, rerandomize_each_iteration=rerandomize_each_iteration, name=name)