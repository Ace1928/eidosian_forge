from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def _sparse_batch(input_dataset, batch_size, row_shape, name=None):
    return _DenseToSparseBatchDataset(input_dataset, batch_size, row_shape, name)