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
def _create_cluster(num_workers, num_ps, has_chief=False, has_eval=False, protocol='grpc', worker_config=None, ps_config=None, eval_config=None, worker_name='worker', ps_name='ps', chief_name='chief'):
    """Creates and starts local servers and returns the cluster_spec dict."""
    worker_ports = [pick_unused_port() for _ in range(num_workers)]
    ps_ports = [pick_unused_port() for _ in range(num_ps)]
    cluster_dict = {}
    if num_workers > 0:
        cluster_dict[worker_name] = ['localhost:%s' % port for port in worker_ports]
    if num_ps > 0:
        cluster_dict[ps_name] = ['localhost:%s' % port for port in ps_ports]
    if has_eval:
        cluster_dict['evaluator'] = ['localhost:%s' % pick_unused_port()]
    if has_chief:
        cluster_dict[chief_name] = ['localhost:%s' % pick_unused_port()]
    cs = server_lib.ClusterSpec(cluster_dict)
    for i in range(num_workers):
        server_lib.Server(cs, job_name=worker_name, protocol=protocol, task_index=i, config=worker_config, start=True)
    for i in range(num_ps):
        server_lib.Server(cs, job_name=ps_name, protocol=protocol, task_index=i, config=ps_config, start=True)
    if has_chief:
        server_lib.Server(cs, job_name=chief_name, protocol=protocol, task_index=0, config=worker_config, start=True)
    if has_eval:
        server_lib.Server(cs, job_name='evaluator', protocol=protocol, task_index=0, config=eval_config, start=True)
    return cluster_dict