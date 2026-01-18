import collections
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export
@tf_export('data.experimental.service.DispatcherConfig')
class DispatcherConfig(collections.namedtuple('DispatcherConfig', ['port', 'protocol', 'work_dir', 'fault_tolerant_mode', 'worker_addresses', 'job_gc_check_interval_ms', 'job_gc_timeout_ms', 'worker_timeout_ms', 'worker_max_concurrent_snapshots'])):
    """Configuration class for tf.data service dispatchers.

  Fields:
    port: Specifies the port to bind to. A value of 0 indicates that the server
      may bind to any available port.
    protocol: The protocol to use for communicating with the tf.data service,
      e.g. "grpc".
    work_dir: A directory to store dispatcher state in. This
      argument is required for the dispatcher to be able to recover from
      restarts.
    fault_tolerant_mode: Whether the dispatcher should write its state to a
      journal so that it can recover from restarts. Dispatcher state, including
      registered datasets and created jobs, is synchronously written to the
      journal before responding to RPCs. If `True`, `work_dir` must also be
      specified.
    worker_addresses: If the job uses auto-sharding, it needs to specify a fixed
      list of worker addresses that will register with the dispatcher. The
      worker addresses should be in the format `"host"` or `"host:port"`, where
      `"port"` is an integer, named port, or `%port%` to match any port.
    job_gc_check_interval_ms: How often the dispatcher should scan through to
      delete old and unused jobs, in milliseconds. If not set, the runtime will
      select a reasonable default. A higher value will reduce load on the
      dispatcher, while a lower value will reduce the time it takes for the
      dispatcher to garbage collect expired jobs.
    job_gc_timeout_ms: How long a job needs to be unused before it becomes a
      candidate for garbage collection, in milliseconds. A value of -1 indicates
      that jobs should never be garbage collected. If not set, the runtime will
      select a reasonable default. A higher value will cause jobs to stay around
      longer with no consumers. This is useful if there is a large gap in
      time between when consumers read from the job. A lower value will reduce
      the time it takes to reclaim the resources from expired jobs.
    worker_timeout_ms: How long to wait for a worker to heartbeat before
      considering it missing. If not set, the runtime will select a reasonable
      default.
    worker_max_concurrent_snapshots: The maximum number of snapshots a worker
      can concurrently process.
  """

    def __new__(cls, port=0, protocol=None, work_dir=None, fault_tolerant_mode=False, worker_addresses=None, job_gc_check_interval_ms=None, job_gc_timeout_ms=None, worker_timeout_ms=None, worker_max_concurrent_snapshots=0):
        if protocol is None:
            protocol = _pywrap_utils.TF_DATA_DefaultProtocol()
        job_gc_check_interval_ms = _get_time_or_placeholder(job_gc_check_interval_ms)
        job_gc_timeout_ms = _get_time_or_placeholder(job_gc_timeout_ms)
        return super().__new__(cls, port, protocol, work_dir, fault_tolerant_mode, worker_addresses, job_gc_check_interval_ms, job_gc_timeout_ms, worker_timeout_ms, worker_max_concurrent_snapshots)