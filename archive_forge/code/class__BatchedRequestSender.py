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
class _BatchedRequestSender:
    """Helper class for building requests that fit under a size limit.

    This class maintains stateful request builders for each of the possible
    request types (scalars, tensors, and blobs).  These accumulate batches
    independently, each maintaining its own byte budget and emitting a request
    when the batch becomes full.  As a consequence, events of different types
    will likely be sent to the backend out of order.  E.g., in the extreme case,
    a single tensor-flavored request may be sent only when the event stream is
    exhausted, even though many more recent scalar events were sent earlier.

    This class is not threadsafe. Use external synchronization if
    calling its methods concurrently.
    """

    def __init__(self, experiment_id, api, allowed_plugins, upload_limits, rpc_rate_limiter, tensor_rpc_rate_limiter, blob_rpc_rate_limiter, tracker):
        self._tag_metadata = {}
        self._allowed_plugins = frozenset(allowed_plugins)
        self._tracker = tracker
        self._scalar_request_sender = _ScalarBatchedRequestSender(experiment_id, api, rpc_rate_limiter, upload_limits.max_scalar_request_size, tracker=self._tracker)
        self._tensor_request_sender = _TensorBatchedRequestSender(experiment_id, api, tensor_rpc_rate_limiter, upload_limits.max_tensor_request_size, upload_limits.max_tensor_point_size, tracker=self._tracker)
        self._blob_request_sender = _BlobRequestSender(experiment_id, api, blob_rpc_rate_limiter, upload_limits.max_blob_request_size, upload_limits.max_blob_size, tracker=self._tracker)
        self._tracker = tracker

    def send_requests(self, run_to_events):
        """Accepts a stream of TF events and sends batched write RPCs.

        Each sent request will be batched, the size of each batch depending on
        the type of data (Scalar vs Tensor vs Blob) being sent.

        Args:
          run_to_events: Mapping from run name to generator of `tf.Event`
            values, as returned by `LogdirLoader.get_run_events`.

        Raises:
          RuntimeError: If no progress can be made because even a single
          point is too large (say, due to a gigabyte-long tag name).
        """
        for run_name, event, value in self._run_values(run_to_events):
            time_series_key = (run_name, value.tag)
            metadata = self._tag_metadata.get(time_series_key)
            first_in_time_series = False
            if metadata is None:
                first_in_time_series = True
                metadata = value.metadata
                self._tag_metadata[time_series_key] = metadata
            plugin_name = metadata.plugin_data.plugin_name
            if value.HasField('metadata') and plugin_name != value.metadata.plugin_data.plugin_name:
                logger.warning('Mismatching plugin names for %s.  Expected %s, found %s.', time_series_key, metadata.plugin_data.plugin_name, value.metadata.plugin_data.plugin_name)
                continue
            if plugin_name not in self._allowed_plugins:
                if first_in_time_series:
                    logger.info('Skipping time series %r with unsupported plugin name %r', time_series_key, plugin_name)
                continue
            if metadata.data_class == summary_pb2.DATA_CLASS_SCALAR:
                self._scalar_request_sender.add_event(run_name, event, value, metadata)
            elif metadata.data_class == summary_pb2.DATA_CLASS_TENSOR:
                self._tensor_request_sender.add_event(run_name, event, value, metadata)
            elif metadata.data_class == summary_pb2.DATA_CLASS_BLOB_SEQUENCE:
                self._blob_request_sender.add_event(run_name, event, value, metadata)
        self._scalar_request_sender.flush()
        self._tensor_request_sender.flush()
        self._blob_request_sender.flush()

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