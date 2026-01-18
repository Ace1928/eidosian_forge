from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_dataset_ops
def _from_tensors(tensors, name):
    return _TensorDataset(tensors, name)