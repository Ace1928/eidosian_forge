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
def _process_watchdog(self):
    """Simulates a cluster management system.

    - If auto_restart is True, it restarts processes that exit with a non-zero
      exit code. Note that when join() times out it overrides auto_restart to
      False.
    - If dependence_on_chief is True, it terminates all processes once the chief
      exits. If auto_restart is also True, it only terminates all processes if
      the chief exit with a zero exit code, otherwise it restarts the chief.

    This runs in self._watchdog_thread.
    """
    while True:
        time.sleep(1)
        with self._process_lock:
            chief = self._processes.get(('chief', 0), None)
            if chief and self._dependence_on_chief and (chief.exitcode is not None):
                if chief.exitcode == 0 or not self._auto_restart:
                    for p in self._processes.values():
                        p.join(timeout=3)
                    self._terminate_all()
                    for p in self._processes.values():
                        p.join()
                    return
            if self._auto_restart:
                has_failure = False
                for (task_type, task_id), p in self._processes.items():
                    if p.exitcode is not None and p.exitcode != 0:
                        has_failure = True
                        logging.info('Restarting failed %s-%d', task_type, task_id)
                        self._start_subprocess_and_reading_thread(task_type, task_id)
                if has_failure:
                    continue
            if all((p.exitcode is not None for p in self._processes.values())):
                return