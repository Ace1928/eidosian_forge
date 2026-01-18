import contextlib
import functools
import time
import grpc
from google.protobuf import message
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.uploader.proto import write_service_pb2
from tensorboard.uploader import logdir_loader
from tensorboard.uploader import upload_tracker
from tensorboard.uploader import util
from tensorboard.backend import process_graph
from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.plugins.graph import metadata as graphs_metadata
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _write_blob_request_iterator(self, blob_sequence_id, seq_index, blob):
    for offset in range(0, len(blob), self._max_blob_request_size):
        chunk = blob[offset:offset + self._max_blob_request_size]
        finalize_object = offset + self._max_blob_request_size >= len(blob)
        request = write_service_pb2.WriteBlobRequest(blob_sequence_id=blob_sequence_id, index=seq_index, data=chunk, offset=offset, crc32c=None, finalize_object=finalize_object, final_crc32c=None, blob_bytes=len(blob))
        yield request