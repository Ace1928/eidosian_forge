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
class MultiWorkerTestBase(test.TestCase):
    """Base class for testing multi node strategy and dataset."""

    @classmethod
    def setUpClass(cls, num_workers=2, num_ps=1):
        """Create a local cluster with 2 workers."""
        cls._cluster_spec = create_in_process_cluster(num_workers=num_workers, num_ps=num_ps)
        cls._default_target = 'grpc://' + cls._cluster_spec['worker'][0]

    def setUp(self):
        self._thread_local = threading.local()
        self._thread_local.cached_session = None
        self._coord = coordinator.Coordinator()

    @contextlib.contextmanager
    def session(self, graph=None, config=None, target=None):
        """Create a test session with master target set to the testing cluster.

    Creates a test session that connects to the local testing cluster.

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      target: the target of session to connect to.

    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """
        config = self._create_config(config)
        if target is None:
            target = self._default_target
        with session.Session(graph=graph, config=config, target=target) as sess:
            yield sess

    @contextlib.contextmanager
    def cached_session(self, graph=None, config=None, target=None):
        """Create a test session with master target set to the testing cluster.

    Creates a test session that connects to the local testing cluster.
    The session is only created once per test and then reused.

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      target: the target of session to connect to.

    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case. Note that the
      session will live until the end of the test.
    """
        config = self._create_config(config)
        if target is None:
            target = self._default_target
        if getattr(self._thread_local, 'cached_session', None) is None:
            self._thread_local.cached_session = session.Session(graph=None, config=config, target=target)
        sess = self._thread_local.cached_session
        with sess.graph.as_default(), sess.as_default():
            yield sess

    def _create_config(self, config):
        if config is None:
            config = config_pb2.ConfigProto(allow_soft_placement=True)
        else:
            config = copy.deepcopy(config)
        config.graph_options.optimizer_options.opt_level = -1
        config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
        return config

    def _run_client(self, client_fn, task_type, task_id, num_gpus, eager_mode, *args, **kwargs):

        def wrapped_client_fn():
            with self._coord.stop_on_exception():
                client_fn(task_type, task_id, num_gpus, *args, **kwargs)
        if eager_mode:
            with context.eager_mode():
                wrapped_client_fn()
        else:
            with context.graph_mode():
                wrapped_client_fn()

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