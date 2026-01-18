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
def _run_between_graph_clients(self, client_fn, cluster_spec, num_gpus, *args, **kwargs):
    """Runs several clients for between-graph replication.

    Args:
      client_fn: a function that needs to accept `task_type`, `task_id`,
        `num_gpus`.
      cluster_spec: a dict specifying jobs in a cluster.
      num_gpus: number of GPUs per worker.
      *args: will be passed to `client_fn`.
      **kwargs: will be passed to `client_fn`.
    """
    threads = []
    for task_type in ['chief', 'worker']:
        for task_id in range(len(cluster_spec.get(task_type, []))):
            t = threading.Thread(target=self._run_client, args=(client_fn, task_type, task_id, num_gpus, context.executing_eagerly()) + args, kwargs=kwargs)
            t.start()
            threads.append(t)
    self._coord.join(threads)