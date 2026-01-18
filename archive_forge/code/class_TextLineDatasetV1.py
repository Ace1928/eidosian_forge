import os
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import from_tensor_slices_op
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['data.TextLineDataset'])
class TextLineDatasetV1(dataset_ops.DatasetV1Adapter):
    """A `Dataset` comprising lines from one or more text files."""

    def __init__(self, filenames, compression_type=None, buffer_size=None, num_parallel_reads=None, name=None):
        wrapped = TextLineDatasetV2(filenames, compression_type, buffer_size, num_parallel_reads, name)
        super(TextLineDatasetV1, self).__init__(wrapped)
    __init__.__doc__ = TextLineDatasetV2.__init__.__doc__

    @property
    def _filenames(self):
        return self._dataset._filenames

    @_filenames.setter
    def _filenames(self, value):
        self._dataset._filenames = value