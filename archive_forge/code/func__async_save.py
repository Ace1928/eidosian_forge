import atexit
import collections
import copy
import queue
import threading
import time
import weakref
from absl import logging
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import UninitializedVariable
from tensorflow.python.ops.variables import Variable
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import object_identity
def _async_save(self):
    """The thread function for the async checkpoint save."""
    with context.executor_scope(executor.new_executor(enable_async=False, enable_streaming_enqueue=False)):
        while self._queue.get():
            logging.info('Starting async checkpoint save on the device: %s', self._default_device)
            async_save_start_time = time.time()
            try:
                with ops.device(self._default_device):
                    with checkpoint_context.async_metrics_context():
                        if self._use_checkpoint_save:
                            self.checkpointer().save(self._save_file_prefix, self._checkpoint_options)
                        else:
                            self.checkpointer()._write(self._save_file_prefix, options=self._checkpoint_options, write_done_callback=self._async_write_done_callback)
            except Exception as e:
                self._async_error = e
            finally:
                self._queue.task_done()
            async_save_end_time = time.time()
            metrics.AddAsyncCheckpointWriteDuration(api_label=_ASYNC_CHECKPOINT, microseconds=_get_duration_microseconds(async_save_start_time, async_save_end_time))
            global _END_TIME_OF_LAST_ASYNC_WRITE
            with _END_TIME_OF_LAST_ASYNC_WRITE_LOCK:
                metrics.AddTrainingTimeSaved(api_label=_ASYNC_CHECKPOINT, microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_ASYNC_WRITE, async_save_start_time))
                _END_TIME_OF_LAST_ASYNC_WRITE = async_save_start_time
    logging.info('Async save thread reached the end of the execution.')