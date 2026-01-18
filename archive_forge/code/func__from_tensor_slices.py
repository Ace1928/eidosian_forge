from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_dataset_ops
def _from_tensor_slices(tensors, name=None):
    return _TensorSliceDataset(tensors, name=name)