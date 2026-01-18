import copy
import json
import os
import threading
import time
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
def _split_cluster_for_evaluator(cluster_spec, task_type):
    """Split the cluster for evaluator since it needn't talk to other tasks."""
    new_cluster_spec = normalize_cluster_spec(cluster_spec).as_dict()
    if task_type == _TaskType.EVALUATOR:
        assert _TaskType.EVALUATOR in new_cluster_spec
        new_cluster_spec = {_TaskType.EVALUATOR: new_cluster_spec[_TaskType.EVALUATOR]}
    else:
        new_cluster_spec.pop(_TaskType.EVALUATOR, None)
    return normalize_cluster_spec(new_cluster_spec)