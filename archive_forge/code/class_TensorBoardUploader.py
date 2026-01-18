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
class TensorBoardUploader:
    """Uploads a TensorBoard logdir to TensorBoard.dev."""

    def __init__(self, writer_client, logdir, allowed_plugins, upload_limits, logdir_poll_rate_limiter=None, rpc_rate_limiter=None, tensor_rpc_rate_limiter=None, blob_rpc_rate_limiter=None, name=None, description=None, verbosity=None, one_shot=None):
        """Constructs a TensorBoardUploader.

        Args:
          writer_client: a TensorBoardWriterService stub instance
          logdir: path of the log directory to upload
          allowed_plugins: collection of string plugin names; events will only
            be uploaded if their time series's metadata specifies one of these
            plugin names
          upload_limits: instance of tensorboard.service.UploadLimits proto.
          logdir_poll_rate_limiter: a `RateLimiter` to use to limit logdir
            polling frequency, to avoid thrashing disks, especially on networked
            file systems
          rpc_rate_limiter: a `RateLimiter` to use to limit write RPC frequency.
            Note this limit applies at the level of single RPCs in the Scalar
            and Tensor case, but at the level of an entire blob upload in the
            Blob case-- which may require a few preparatory RPCs and a stream
            of chunks.  Note the chunk stream is internally rate-limited by
            backpressure from the server, so it is not a concern that we do not
            explicitly rate-limit within the stream here.
          name: String name to assign to the experiment.
          description: String description to assign to the experiment.
          verbosity: Level of verbosity, an integer. Supported value:
              0 - No upload statistics is printed.
              1 - Print upload statistics while uploading data (default).
         one_shot: Once uploading starts, upload only the existing data in
            the logdir and then return immediately, instead of the default
            behavior of continuing to listen for new data in the logdir and
            upload them when it appears.
        """
        self._api = writer_client
        self._logdir = logdir
        self._allowed_plugins = frozenset(allowed_plugins)
        self._upload_limits = upload_limits
        self._name = name
        self._description = description
        self._verbosity = 1 if verbosity is None else verbosity
        self._one_shot = False if one_shot is None else one_shot
        self._request_sender = None
        self._experiment_id = None
        if logdir_poll_rate_limiter is None:
            self._logdir_poll_rate_limiter = util.RateLimiter(_MIN_LOGDIR_POLL_INTERVAL_SECS)
        else:
            self._logdir_poll_rate_limiter = logdir_poll_rate_limiter
        if rpc_rate_limiter is None:
            self._rpc_rate_limiter = util.RateLimiter(self._upload_limits.min_scalar_request_interval / 1000)
        else:
            self._rpc_rate_limiter = rpc_rate_limiter
        if tensor_rpc_rate_limiter is None:
            self._tensor_rpc_rate_limiter = util.RateLimiter(self._upload_limits.min_tensor_request_interval / 1000)
        else:
            self._tensor_rpc_rate_limiter = tensor_rpc_rate_limiter
        if blob_rpc_rate_limiter is None:
            self._blob_rpc_rate_limiter = util.RateLimiter(self._upload_limits.min_blob_request_interval / 1000)
        else:
            self._blob_rpc_rate_limiter = blob_rpc_rate_limiter
        active_filter = lambda secs: secs + _EVENT_FILE_INACTIVE_SECS >= time.time()
        directory_loader_factory = functools.partial(directory_loader.DirectoryLoader, loader_factory=event_file_loader.TimestampedEventFileLoader, path_filter=io_wrapper.IsTensorFlowEventsFile, active_filter=active_filter)
        self._logdir_loader = logdir_loader.LogdirLoader(self._logdir, directory_loader_factory)
        self._tracker = upload_tracker.UploadTracker(verbosity=self._verbosity, one_shot=self._one_shot)

    def has_data(self) -> bool:
        """Returns this object's upload tracker."""
        return self._tracker.has_data()

    @property
    def experiment_id(self) -> str:
        """Returns the experiment_id associated with this uploader.

        May be none if no experiment is set, for instance, if
        `create_experiment` has not been called.
        """
        return self._experiment_id

    def create_experiment(self):
        """Creates an Experiment for this upload session and returns the ID."""
        logger.info('Creating experiment')
        request = write_service_pb2.CreateExperimentRequest(name=self._name, description=self._description)
        response = grpc_util.call_with_retries(self._api.CreateExperiment, request)
        self._request_sender = _BatchedRequestSender(response.experiment_id, self._api, allowed_plugins=self._allowed_plugins, upload_limits=self._upload_limits, rpc_rate_limiter=self._rpc_rate_limiter, tensor_rpc_rate_limiter=self._tensor_rpc_rate_limiter, blob_rpc_rate_limiter=self._blob_rpc_rate_limiter, tracker=self._tracker)
        self._experiment_id = response.experiment_id
        return response.experiment_id

    def start_uploading(self):
        """Uploads data from the logdir.

        This will continuously scan the logdir, uploading as data is added
        unless the uploader was built with the _one_shot option, in which
        case it will terminate after the first scan.

        Raises:
          RuntimeError: If `create_experiment` has not yet been called.
          ExperimentNotFoundError: If the experiment is deleted during the
            course of the upload.
        """
        if self._request_sender is None:
            raise RuntimeError('Must call create_experiment() before start_uploading()')
        while True:
            self._logdir_poll_rate_limiter.tick()
            self._upload_once()
            if self._one_shot:
                break

    def _upload_once(self):
        """Runs one upload cycle, sending zero or more RPCs."""
        logger.info('Starting an upload cycle')
        sync_start_time = time.time()
        self._logdir_loader.synchronize_runs()
        sync_duration_secs = time.time() - sync_start_time
        logger.info('Logdir sync took %.3f seconds', sync_duration_secs)
        run_to_events = self._logdir_loader.get_run_events()
        with self._tracker.send_tracker():
            self._request_sender.send_requests(run_to_events)