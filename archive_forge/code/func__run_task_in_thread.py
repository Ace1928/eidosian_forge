import contextlib
import copy
import json
import os
import subprocess
import sys
import threading
import unittest
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import server_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _run_task_in_thread(self, task_fn, cluster_spec, task_type, task_id, *args, **kwargs):
    """Run tasks in a thread.

    If `tf_config` is provided, use it for the new thread; if not, construct one
    from `cluster_spec`, `task_type`, and `task_id`, and provide it to the new
    thread to be set as `TF_CONFIG` environment.

    Args:
      task_fn: The function to run in the new thread.
      cluster_spec: The cluster spec.
      task_type: The task type.
      task_id: The task id.
      *args: Additional positional arguments to provide to the thread's task_fn.
      **kwargs: Additional keyword arguments to provide to the thread's task_fn.
        If `tf_config` is provided, that dict will be used for the TF_CONFIG for
        the new thread.

    Returns:
      The thread that has started.
    """
    tf_config = kwargs.pop('tf_config', None)
    if tf_config is None:
        if task_type:
            tf_config = {'cluster': cluster_spec, 'task': {'type': task_type, 'index': task_id}}
        else:
            tf_config = {'cluster': cluster_spec}
    t = threading.Thread(target=self._task_thread, args=(task_fn, tf_config, context.executing_eagerly()) + args, kwargs=kwargs)
    t.start()
    return t