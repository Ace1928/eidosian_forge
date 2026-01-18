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
def _run_values(self, run_to_events):
    """Helper generator to create a single stream of work items.

        Note that `dataclass_compat` may emit multiple variants of
        the same event, for backwards compatibility.  Thus this stream should
        be filtered to obtain the desired version of each event.  Here, we
        ignore any event that does not have a `summary` field.

        Furthermore, the events emitted here could contain values that do not
        have `metadata.data_class` set; these too should be ignored.  In
        `_send_summary_value(...)` above, we switch on `metadata.data_class`
        and drop any values with an unknown (i.e., absent or unrecognized)
        `data_class`.
        """
    for run_name, events in run_to_events.items():
        for event in events:
            _filter_graph_defs(event)
            for value in event.summary.value:
                yield (run_name, event, value)