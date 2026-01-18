from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_experimental_dataset_ops
def _unique(input_dataset, name):
    return _UniqueDataset(input_dataset, name)