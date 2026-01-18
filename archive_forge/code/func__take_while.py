from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def _take_while(input_dataset, predicate, name=None):
    """See `Dataset.take_while()` for details."""
    return _TakeWhileDataset(input_dataset, predicate, name=name)