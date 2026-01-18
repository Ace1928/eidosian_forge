from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import directed_interleave_op
from tensorflow.python.data.ops import map_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.types import data as data_types
def _skip_datasets_with_zero_weight(datasets, weights):
    datasets_and_weights = [(dataset, weight) for dataset, weight in zip(datasets, weights) if weight > 0]
    return zip(*datasets_and_weights) if datasets_and_weights else ([datasets[0].take(0)], [1.0])