import threading
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
class WatchdogManager(threading.Thread):
    """Configures worker watchdog timer and handles periodic pings.

  Usage:
    # Ping workers every minute, shutting down workers if they haven't received
    # a ping after 1 hour.
    watchdog_manager = WatchdogManager(
      ping_interval=60, shutdown_timeout=3600
    )

    # Use as a context manager, resetting watchdog on context exit:
    with watchdog_manager:
      session.run(...)

    # Or setup globally; watchdog will remain active until program exit.
    watchdog_manager.configure_and_run()
  """

    def __init__(self, session, devices=None, ping_interval=60, shutdown_timeout=2 * 3600):
        """Initialize a watchdog manager.

    Args:
      session: Session connected to worker devices.  A cloned session and graph
        will be created for managing worker pings.
      devices: Set of devices to monitor.  If none, all workers will be
        monitored.
      ping_interval: Time, in seconds, between watchdog pings.
      shutdown_timeout: Time, in seconds, before watchdog timeout.
    """
        threading.Thread.__init__(self)
        self.ping_interval = ping_interval
        self.shutdown_timeout = shutdown_timeout
        self.daemon = True
        self._config = session._config
        self._target = session.sess_str
        self._running = False
        self._devices = devices
        self._graph = None
        self._session = None
        self._worker_manager = None

    def _reset_manager(self, stopping=False):
        """Reset the graph, session and worker manager."""
        self._graph = ops.Graph()
        self._session = session_lib.Session(target=self._target, graph=self._graph, config=self._config)
        if self._devices is None:
            self._devices = all_worker_devices(self._session)
        with self._graph.as_default():
            self._worker_manager = WorkerHeartbeatManager.from_devices(self._session, self._devices)
        if stopping:
            timeout_ms = -1
            shutdown_mode = event_pb2.NOT_CONFIGURED
        else:
            timeout_ms = self.shutdown_timeout * 1000
            shutdown_mode = event_pb2.WAIT_FOR_COORDINATOR
        self._worker_manager.configure(event_pb2.WorkerHeartbeatRequest(watchdog_config=event_pb2.WatchdogConfig(timeout_ms=timeout_ms), shutdown_mode=shutdown_mode))

    def configure_and_run(self):
        logging.info('Enabling watchdog timer with %d second timeout and %d second ping interval.', self.shutdown_timeout, self.ping_interval)
        self._reset_manager()
        self._running = True
        self.start()

    def stop(self):
        logging.info('Stopping worker watchdog.')
        self._reset_manager(stopping=True)
        self._running = False
        self.join()

    def __enter__(self):
        self.configure_and_run()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        while self._running:
            try:
                self._worker_manager.ping(request=None)
                time.sleep(self.ping_interval)
            except errors.OpError as e:
                logging.debug('Caught error while sending heartbeat: %s', e)
                self._reset_manager()