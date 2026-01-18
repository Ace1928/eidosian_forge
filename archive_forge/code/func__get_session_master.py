from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import os
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _get_session_master(cluster_spec, task_type, task_id, tf_config):
    """Returns the appropriate address for TensorFlow master.

  The order of precedence to determine the TF session master is as follows:
  1. If `tf_session_master` is set in TF_CONFIG environment variable, takes it.
  2. If the cluster has only one node, returns empty string ''.
  3. Returns the grpc address according to the task type and id in the cluster.
     This is between-graph replication.

  Note: task_type and task_id must be validated. Typically, validated using
  `_validate_task_type_and_task_id`.

  Args:
    cluster_spec: A `ClusterSpec` instance.
    task_type: String. Task type for current node.
    task_id: Int. Task id for current node.
    tf_config: Dict. Python dict for the TF_CONFIG environment variable.

  Raises:
    RuntimeError: If `cluster_spec` is not set.

  """
    if _SESSION_MASTER_KEY in tf_config:
        return tf_config[_SESSION_MASTER_KEY]
    if not cluster_spec:
        raise RuntimeError('Internal error: `_get_session_master` does not expect empty cluster_spec.')
    jobs = cluster_spec.jobs
    if len(jobs) == 1 and len(cluster_spec.job_tasks(jobs[0])) == 1:
        return _LOCAL_MASTER
    addresses = cluster_spec.job_tasks(task_type)
    return _GRPC_SCHEME + addresses[task_id]