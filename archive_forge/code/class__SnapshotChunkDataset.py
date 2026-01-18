import multiprocessing
import os
from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.protobuf import snapshot_pb2
from tensorflow.python.data.experimental.service import _pywrap_snapshot_utils
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import nested_structure_coder
class _SnapshotChunkDataset(dataset_ops.DatasetSource):
    """A dataset for one chunk file from a tf.data distributed snapshot."""

    def __init__(self, chunk_file, element_spec, compression):
        self._chunk_file = chunk_file
        self._element_spec = element_spec
        variant_tensor = ged_ops.snapshot_chunk_dataset(chunk_file, compression=compression, **self._flat_structure)
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        return self._element_spec