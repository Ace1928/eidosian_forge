import collections
import contextlib
import json
import os
import signal
import sys
import threading
import time
import unittest
import weakref
from absl import logging
import six
from six.moves import queue as Queue
from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.util.tf_export import tf_export
class _ProcFunc(object):
    """Represents a callable to run in a subprocess."""

    @contextlib.contextmanager
    def _runtime_mode(self, executing_eagerly):
        if executing_eagerly:
            with context.eager_mode():
                yield
        else:
            with context.graph_mode():
                yield

    def _message_checking_func(self, task_type, task_id):
        """A function that regularly checks messages from parent process."""
        while True:
            try:
                message = self._resources.parent_to_sub_queue.get(block=False)
                if not message.startswith('terminate'):
                    raise ValueError('Unrecognized message: {}'.format(message))
                if message == 'terminate {} {}'.format(task_type, task_id):
                    break
                else:
                    self._resources.parent_to_sub_queue.put(message)
                    time.sleep(1)
            except Queue.Empty:
                time.sleep(0.1)
        self._resources.process_status_queue.put(_ProcessStatusInfo(task_type=task_type, task_id=task_id, is_successful=True, exc_info=None, return_value=None))
        os._exit(1)

    def _close_streaming(self):
        """Close stdout, stderr and streaming pipe.

    We need to explicitly close them since Tensorflow may take a while to exit,
    so that the reading threads in the main process can exit more quickly.
    """
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.close()
        sys.stderr.close()
        self._resources.streaming_pipe_w.close()

    def __call__(self, resources, test_env, fn, args, kwargs, use_dill_for_args):
        """The wrapper function that actually gets run in child process(es)."""
        global _barrier
        self._resources = resources
        _barrier = self._resources.barrier
        fn = dill.loads(fn)
        if use_dill_for_args:
            args = dill.loads(args)
            kwargs = dill.loads(kwargs)
        if faulthandler is not None:
            faulthandler.enable()
            faulthandler.register(signal.SIGTERM, chain=True)
        logging.set_stderrthreshold(logging.DEBUG)
        os.dup2(resources.streaming_pipe_w.fileno(), sys.stdout.fileno())
        os.dup2(resources.streaming_pipe_w.fileno(), sys.stderr.fileno())
        pid = os.getpid()
        logging.info('Subprocess with PID %d (%s, %d) is now being started.', pid, test_env.task_type, test_env.task_id)
        logging.info('TF_CONFIG: %r', os.environ['TF_CONFIG'])
        threading.Thread(target=self._message_checking_func, args=(test_env.task_type, test_env.task_id), daemon=True).start()
        if test_env.v2_enabled:
            v2_compat.enable_v2_behavior()
        with self._runtime_mode(test_env.executing_eagerly):
            info = _run_contained(test_env.task_type, test_env.task_id, fn, args, kwargs)
            self._resources.process_status_queue.put(info)
            if not info.is_successful:
                six.reraise(*info.exc_info)
            self._close_streaming()
        sys.exit(0)