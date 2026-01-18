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
def _create_tag(self, run_proto, tag_name, metadata):
    """Adds a tag for the given value, if there's space.

        Args:
          run_proto: `WriteTensorRequest.Run` proto to which to add a tag.
          tag_name: String name of the tag to add (as `value.tag`).
          metadata: TensorBoard `SummaryMetadata` proto from the first
            occurrence of this time series.

        Returns:
          The `WriteTensorRequest.Tag` that was added to `run_proto.tags`.

        Raises:
          _OutOfSpaceError: If adding the tag would exceed the remaining
            request budget.
        """
    tag_proto = run_proto.tags.add(name=tag_name)
    tag_proto.metadata.CopyFrom(metadata)
    self._byte_budget_manager.add_tag(tag_proto)
    return tag_proto