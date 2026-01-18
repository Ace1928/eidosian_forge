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
def _run_single_worker(worker_fn, strategy, cluster_spec, task_type, task_id, session_config, rpc_layer='', worker_barrier=None, coord=None):
    """Runs a single worker by calling `worker_fn` under context."""
    session_config = copy.deepcopy(session_config)
    strategy = copy.deepcopy(strategy)
    if task_type == _TaskType.EVALUATOR:
        if strategy:
            strategy.configure(session_config)
    else:
        assert strategy
        strategy.configure(session_config, cluster_spec, task_type, task_id)
    context = _WorkerContext(strategy, cluster_spec, task_type, task_id, session_config=session_config, rpc_layer=rpc_layer, worker_barrier=worker_barrier)
    with context:
        if coord:
            with coord.stop_on_exception():
                return worker_fn(strategy)
        else:
            return worker_fn(strategy)