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
def _varint_cost(n):
    """Computes the size of `n` encoded as an unsigned base-128 varint.

    This should be consistent with the proto wire format:
    <https://developers.google.com/protocol-buffers/docs/encoding#varints>

    Args:
      n: A non-negative integer.

    Returns:
      An integer number of bytes.
    """
    result = 1
    while n >= 128:
        result += 1
        n >>= 7
    return result