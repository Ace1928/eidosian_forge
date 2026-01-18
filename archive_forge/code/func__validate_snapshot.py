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
def _validate_snapshot(path, metadata, element_spec, compression):
    """Validates a tf.data distributed snapshot.

  Args:
    path: Root path of the distributed snapshot.
    metadata: The DistributedSnapshotMetadata of the snapshot.
    element_spec: Dataset element_spec.
    compression: Compression method used for saving.

  Raises:
    ValueError if the snapshot is invalid.
  """
    if not gfile.Exists(path):
        raise ValueError(f'Failed to load tf.data snapshot at {path}: The snapshot directory does not exist.')
    error_file = _pywrap_snapshot_utils.TF_DATA_SnapshotErrorFilePath(path)
    if gfile.Exists(error_file):
        with gfile.GFile(error_file, 'r') as f:
            raise ValueError(f'Failed to load tf.data snapshot at {path}. The save job failed to write it. Status: {f.read()}')
    done_file = _pywrap_snapshot_utils.TF_DATA_SnapshotDoneFilePath(path)
    if not gfile.Exists(done_file):
        raise ValueError(f'Failed to load tf.data snapshot at {path}. The save job has not finished writing the snapshot.')
    snapshot_element_spec = _parse_element_spec(metadata.element_spec)
    if element_spec and element_spec != snapshot_element_spec:
        raise ValueError(f'Failed to load tf.data snapshot at {path}. User specified element_spec {element_spec}, but the actual element_spec is {snapshot_element_spec}.')
    if compression and compression != metadata.compression:
        raise ValueError(f'Failed to load tf.data snapshot at {path}. User specified compression {compression}, but the actual compression is {metadata.compression}.')