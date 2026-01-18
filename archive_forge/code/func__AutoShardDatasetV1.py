from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.tf_export import tf_export
def _AutoShardDatasetV1(input_dataset, num_workers, index, num_replicas=None):
    return dataset_ops.DatasetV1Adapter(_AutoShardDataset(input_dataset, num_workers, index, num_replicas))