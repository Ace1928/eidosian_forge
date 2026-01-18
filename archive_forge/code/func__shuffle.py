from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
def _shuffle(input_dataset, buffer_size, seed=None, reshuffle_each_iteration=None, name=None):
    return _ShuffleDataset(input_dataset, buffer_size, seed, reshuffle_each_iteration, name=name)