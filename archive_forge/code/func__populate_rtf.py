import contextlib
import grpc
from tensorboard.util import tensor_util
from tensorboard.util import timing
from tensorboard import errors
from tensorboard.data import provider
from tensorboard.data.proto import data_provider_pb2
from tensorboard.data.proto import data_provider_pb2_grpc
def _populate_rtf(run_tag_filter, rtf_proto):
    """Copies `run_tag_filter` into `rtf_proto`."""
    if run_tag_filter is None:
        return
    if run_tag_filter.runs is not None:
        rtf_proto.runs.names[:] = sorted(run_tag_filter.runs)
    if run_tag_filter.tags is not None:
        rtf_proto.tags.names[:] = sorted(run_tag_filter.tags)