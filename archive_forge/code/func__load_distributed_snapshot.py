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
def _load_distributed_snapshot(path, metadata, reader_func):
    """Loads a distributed snapshot."""
    chunks_dir = _pywrap_snapshot_utils.TF_DATA_CommittedChunksDirectory(path)
    chunk_files = [os.path.join(chunks_dir, f) for f in gfile.ListDirectory(chunks_dir)]
    dataset = dataset_ops.Dataset.from_tensor_slices(chunk_files)
    dataset = dataset.map(lambda chunk_file: _SnapshotChunkDataset(chunk_file, element_spec=_parse_element_spec(metadata.element_spec), compression=metadata.compression))
    return reader_func(dataset)