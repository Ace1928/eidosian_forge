from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
def _cache(input_dataset, filename, name):
    return CacheDataset(input_dataset, filename, name)