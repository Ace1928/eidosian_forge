import gc
import os
import sys
import threading
import time
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator as thread_coordinator
from tensorflow.python.training import server_lib
def _ensure_threads_closed(self):
    """Ensures worker and preemption threads are closed."""
    running_threads = test_util.get_running_threads()
    self.assertTrue(test_util.has_thread(_WORKER_THREAD_PREFIX, running_threads))
    self.assertIn(_WORKER_PREEMPTION_THREAD_NAME, running_threads)
    if sys.getrefcount(self.cluster_coord) > 2:
        try:
            test_util.show_backref(self.cluster_coord)
        except:
            pass
    self.cluster_coord = None
    self.strategy = None
    gc.collect()
    time.sleep(1)
    running_threads = test_util.get_running_threads()
    self.assertNotIn(_WORKER_PREEMPTION_THREAD_NAME, running_threads)
    self.assertFalse(test_util.has_thread(_WORKER_THREAD_PREFIX, running_threads), 'Worker thread is not stopped properly.')