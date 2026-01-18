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
class GracefulShutdownHook(session_run_hook.SessionRunHook):
    """Session hook that watches for shutdown events.

  If a shutdown is indicated, `saver.save(checkpoint_prefix)` is executed, and a
  SystemShutdown exception is raised to terminate the main session.  If `saver`
  is None the `SAVERS` collection will be read to find a saver.

  `on_shutdown_hooks` is an optional list of functions that should be called
  after checkpointing.  The function is called with (`run_context`,
  `all_workers`, `lame_workers`).

  If `heartbeat_group` is not specified, it will default to all CPU workers
  in the system.
  """

    def __init__(self, checkpoint_prefix, saver=None, on_shutdown_hooks=None):
        self._saver = saver
        self._checkpoint_prefix = checkpoint_prefix
        self._on_shutdown_hooks = on_shutdown_hooks if on_shutdown_hooks else []
        self._graph = ops.Graph()
        self._workers = None
        self._session = None
        self._heartbeat_supported = False

    def after_create_session(self, training_session, coord):
        if training_util.get_global_step() is None and self.saver() is not None:
            raise ValueError('Saver defined but no global step.  Run `get_or_create_global_step()` in your model definition to allow checkpointing.')
        with self._graph.as_default():
            logging.info('Installing graceful shutdown hook.')
            self._session = _clone_session(training_session, self._graph)
            self._workers = WorkerHeartbeatManager.from_devices(self._session, all_worker_devices(self._session))
            self._heartbeat_supported = self._workers.num_workers() > 0
            if self._heartbeat_supported:
                try:
                    self._workers.configure(event_pb2.WorkerHeartbeatRequest(shutdown_mode=event_pb2.WAIT_FOR_COORDINATOR))
                except errors.InvalidArgumentError:
                    logging.warn('TPU device does not support heartbeats. Failure handling will be disabled.')
                    self._heartbeat_supported = False
            else:
                logging.warn('No workers support heartbeats. Failure handling will be disabled.')

    def saver(self):
        if self._saver:
            return self._saver
        savers = ops.get_collection(ops.GraphKeys.SAVERS)
        if not savers:
            return None
        if not isinstance(savers, list):
            return savers
        if len(savers) > 1:
            logging.error('Multiple savers in the SAVERS collection.  On-demand checkpointing will be disabled. Pass an explicit `saver` to the constructor to override this behavior.')
            return None
        return savers[0]

    def after_run(self, run_context, run_values):
        del run_values
        if not self._heartbeat_supported:
            return
        lame_workers = self._workers.lame_workers()
        if lame_workers:
            logging.info('ShutdownHook: lame workers found: %s', lame_workers)
            if self.saver():
                logging.info('ShutdownHook: saving checkpoint to %s', self._checkpoint_prefix)
                self.saver().save(run_context.session, self._checkpoint_prefix, global_step=training_util.get_global_step(), write_state=True)
            else:
                logging.info('ShutdownHook: no Saver defined.')
            for fn in self._on_shutdown_hooks:
                fn(run_context, self._workers, lame_workers)