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
def _task_thread(self, task_fn, tf_config, executing_eagerly, *args, **kwargs):
    with self._coord.stop_on_exception():
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
        if executing_eagerly:
            with context.eager_mode():
                task_fn(*args, **kwargs)
        else:
            with ops.Graph().as_default(), context.graph_mode():
                task_fn(*args, **kwargs)