import functools
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import random_op
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, 'Use `tf.data.Dataset.random(...)`.')
@tf_export(v1=['data.experimental.RandomDataset'])
class RandomDatasetV1(dataset_ops.DatasetV1Adapter):
    """A `Dataset` of pseudorandom values."""

    @functools.wraps(RandomDatasetV2.__init__)
    def __init__(self, seed=None):
        wrapped = RandomDatasetV2(seed)
        super(RandomDatasetV1, self).__init__(wrapped)