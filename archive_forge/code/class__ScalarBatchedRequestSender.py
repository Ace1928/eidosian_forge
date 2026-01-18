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
class _ScalarBatchedRequestSender:
    """Helper class for building requests that fit under a size limit.

    This class accumulates a current request.  `add_event(...)` may or may not
    send the request (and start a new one).  After all `add_event(...)` calls
    are complete, a final call to `flush()` is needed to send the final request.

    This class is not threadsafe. Use external synchronization if calling its
    methods concurrently.
    """

    def __init__(self, experiment_id, api, rpc_rate_limiter, max_request_size, tracker):
        if experiment_id is None:
            raise ValueError('experiment_id cannot be None')
        self._experiment_id = experiment_id
        self._api = api
        self._rpc_rate_limiter = rpc_rate_limiter
        self._byte_budget_manager = _ByteBudgetManager(max_request_size)
        self._tracker = tracker
        self._runs = {}
        self._tags = {}
        self._new_request()

    def _new_request(self):
        """Allocates a new request and refreshes the budget."""
        self._request = write_service_pb2.WriteScalarRequest()
        self._runs.clear()
        self._tags.clear()
        self._num_values = 0
        self._request.experiment_id = self._experiment_id
        self._byte_budget_manager.reset(self._request)

    def add_event(self, run_name, event, value, metadata):
        """Attempts to add the given event to the current request.

        If the event cannot be added to the current request because the byte
        budget is exhausted, the request is flushed, and the event is added
        to the next request.
        """
        try:
            self._add_event_internal(run_name, event, value, metadata)
        except _OutOfSpaceError:
            self.flush()
            try:
                self._add_event_internal(run_name, event, value, metadata)
            except _OutOfSpaceError:
                raise RuntimeError('add_event failed despite flush')

    def _add_event_internal(self, run_name, event, value, metadata):
        run_proto = self._runs.get(run_name)
        if run_proto is None:
            run_proto = self._create_run(run_name)
            self._runs[run_name] = run_proto
        tag_proto = self._tags.get((run_name, value.tag))
        if tag_proto is None:
            tag_proto = self._create_tag(run_proto, value.tag, metadata)
            self._tags[run_name, value.tag] = tag_proto
        self._create_point(tag_proto, event, value)
        self._num_values += 1

    def flush(self):
        """Sends the active request after removing empty runs and tags.

        Starts a new, empty active request.
        """
        request = self._request
        _prune_empty_tags_and_runs(request)
        if not request.runs:
            return
        self._rpc_rate_limiter.tick()
        with _request_logger(request, request.runs), self._tracker.scalars_tracker(self._num_values):
            try:
                grpc_util.call_with_retries(self._api.WriteScalar, request)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.NOT_FOUND:
                    raise ExperimentNotFoundError()
                logger.error('Upload call failed with error %s', e)
        self._new_request()

    def _create_run(self, run_name):
        """Adds a run to the live request, if there's space.

        Args:
          run_name: String name of the run to add.

        Returns:
          The `WriteScalarRequest.Run` that was added to `request.runs`.

        Raises:
          _OutOfSpaceError: If adding the run would exceed the remaining
            request budget.
        """
        run_proto = self._request.runs.add(name=run_name)
        self._byte_budget_manager.add_run(run_proto)
        return run_proto

    def _create_tag(self, run_proto, tag_name, metadata):
        """Adds a tag for the given value, if there's space.

        Args:
          run_proto: `WriteScalarRequest.Run` proto to which to add a tag.
          tag_name: String name of the tag to add (as `value.tag`).
          metadata: TensorBoard `SummaryMetadata` proto from the first
            occurrence of this time series.

        Returns:
          The `WriteScalarRequest.Tag` that was added to `run_proto.tags`.

        Raises:
          _OutOfSpaceError: If adding the tag would exceed the remaining
            request budget.
        """
        tag_proto = run_proto.tags.add(name=tag_name)
        tag_proto.metadata.CopyFrom(metadata)
        self._byte_budget_manager.add_tag(tag_proto)
        return tag_proto

    def _create_point(self, tag_proto, event, value):
        """Adds a scalar point to the given tag, if there's space.

        Args:
          tag_proto: `WriteScalarRequest.Tag` proto to which to add a point.
          event: Enclosing `Event` proto with the step and wall time data.
          value: Scalar `Summary.Value` proto with the actual scalar data.

        Raises:
          _OutOfSpaceError: If adding the point would exceed the remaining
            request budget.
        """
        point = tag_proto.points.add()
        point.step = event.step
        point.value = tensor_util.make_ndarray(value.tensor).item()
        util.set_timestamp(point.wall_time, event.wall_time)
        try:
            self._byte_budget_manager.add_point(point)
        except _OutOfSpaceError:
            tag_proto.points.pop()
            raise