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
def _validate_task_type_and_task_id(cluster_spec, task_env, chief_task_type):
    """Validates the task type and index in `task_env` according to cluster."""
    if chief_task_type not in cluster_spec.jobs:
        raise ValueError('If "cluster" is set in TF_CONFIG, it must have one "%s" node.' % chief_task_type)
    if len(cluster_spec.job_tasks(chief_task_type)) > 1:
        raise ValueError('The "cluster" in TF_CONFIG must have only one "%s" node.' % chief_task_type)
    task_type = task_env.get(_TASK_TYPE_KEY, None)
    task_id = task_env.get(_TASK_ID_KEY, None)
    if not task_type:
        raise ValueError('If "cluster" is set in TF_CONFIG, task type must be set.')
    if task_id is None:
        raise ValueError('If "cluster" is set in TF_CONFIG, task index must be set.')
    task_id = int(task_id)
    if task_id < 0:
        raise ValueError('Task index must be non-negative number.')
    if task_type == TaskType.EVALUATOR:
        return (task_type, task_id)
    if task_type not in cluster_spec.jobs:
        raise ValueError('%s is not a valid task_type in the cluster_spec:\n%s\n\nNote that these values may be coming from the TF_CONFIG environment variable.' % (task_type, cluster_spec))
    addresses = cluster_spec.job_tasks(task_type)
    if not 0 <= task_id < len(addresses):
        raise ValueError('%d is not a valid task_id for task_type %s in the cluster_spec:\n%s\n\nNote that these values may be coming from the TF_CONFIG environment variable.' % (task_id, task_type, cluster_spec))
    return (task_type, task_id)