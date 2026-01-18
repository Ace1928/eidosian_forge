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
class WorkerHeartbeatManager(object):
    """Manages the status/heartbeat monitor for a set of workers."""

    def __init__(self, session, devices, heartbeat_ops, request_placeholder):
        """Construct a new WorkerHeartbeatManager.

    (Prefer using `WorkerHeartbeatManager.from_devices` when possible.)

    Args:
      session: `tf.compat.v1.Session`, session to use for heartbeat operations.
      devices: `list[string]` Set of devices to connect to.
      heartbeat_ops: `list[tf.Operation]` Heartbeat operations.
      request_placeholder: `tf.Placeholder[String]` Placeholder used to specify
        the WorkerHeartbeatRequest protocol buffer.
    """
        self._session = session
        self._devices = devices
        self._ops = heartbeat_ops
        self._request_placeholder = request_placeholder

    @staticmethod
    def from_devices(session, devices):
        """Construct a heartbeat manager for the given devices."""
        if not devices:
            logging.error('Trying to create heartbeat manager with no devices?')
        logging.info('Creating heartbeat manager for %s', devices)
        request_placeholder = array_ops.placeholder(name='worker_heartbeat_request', dtype=dtypes.string)
        heartbeat_ops = []
        for device in devices:
            with ops.device(device):
                heartbeat_ops.append(tpu_ops.worker_heartbeat(request_placeholder))
        return WorkerHeartbeatManager(session, devices, heartbeat_ops, request_placeholder)

    def num_workers(self):
        return len(self._devices)

    def configure(self, message):
        """Configure heartbeat manager for all devices.

    Args:
      message: `event_pb2.WorkerHeartbeatRequest`
    Returns: `None`
    """
        logging.info('Configuring worker heartbeat: %s', text_format.MessageToString(message))
        self._session.run(self._ops, {self._request_placeholder: message.SerializeToString()})

    def ping(self, request=None, timeout_in_ms=60000):
        """Ping all workers, returning the parsed status results."""
        if request is None:
            request = event_pb2.WorkerHeartbeatRequest()
        options = config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
        results = self._session.run(self._ops, feed_dict={self._request_placeholder: request.SerializeToString()}, options=options)
        parsed_results = [event_pb2.WorkerHeartbeatResponse.FromString(res_pb) for res_pb in results]
        logging.debug('Ping results: %s', parsed_results)
        return parsed_results

    def lame_workers(self):
        """Ping all workers, returning manager containing lame workers (or None)."""
        ping_results = self.ping()
        lame_workers = []
        for ping_response, device, op in zip(ping_results, self._devices, self._ops):
            if ping_response.health_status != event_pb2.OK:
                lame_workers.append((device, op))
        if not lame_workers:
            return None
        bad_devices, bad_ops = zip(*lame_workers)
        return WorkerHeartbeatManager(self._session, bad_devices, bad_ops, self._request_placeholder)

    def __repr__(self):
        return 'HeartbeatManager(%s)' % ','.join(self._devices)

    def shutdown(self, wait_time_in_ms=60000, exit_code=0):
        """Shutdown all workers after `shutdown_timeout_secs`."""
        logging.info('Shutting down %s.', self)
        req = event_pb2.WorkerHeartbeatRequest(watchdog_config=event_pb2.WatchdogConfig(timeout_ms=wait_time_in_ms), shutdown_mode=event_pb2.SHUTDOWN_AFTER_TIMEOUT, exit_code=event_pb2.RequestedExitCode(exit_code=exit_code))
        self.configure(req)
        sleep_sec = 10.0 + wait_time_in_ms / 1000
        logging.info('Waiting %.2f seconds for worker shutdown.', sleep_sec)
        time.sleep(sleep_sec)