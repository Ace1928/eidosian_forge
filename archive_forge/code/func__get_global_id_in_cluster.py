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
def _get_global_id_in_cluster(cluster_spec, task_type, task_id, chief_task_type):
    """Returns the global id in cluster."""
    task_type_ordered_list = [chief_task_type]
    task_type_ordered_list.extend([t for t in sorted(cluster_spec.jobs) if t != chief_task_type and t != TaskType.PS])
    if TaskType.PS in cluster_spec.jobs:
        task_type_ordered_list.append(TaskType.PS)
    next_global_id = 0
    for t in task_type_ordered_list:
        if t == task_type:
            return next_global_id + task_id
        next_global_id += len(cluster_spec.job_tasks(t))
    raise RuntimeError('Internal Error: `task_type` ({}) is not in cluster_spec ({}).'.format(task_type, cluster_spec))