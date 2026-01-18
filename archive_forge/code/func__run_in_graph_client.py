import copy
import json
import os
import threading
import time
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
def _run_in_graph_client(worker_fn, strategy, eval_fn, eval_strategy, cluster_spec, session_config, rpc_layer):
    """Runs a standalone client for in-graph replication."""
    coord = coordinator.Coordinator()
    eval_thread = None
    if _TaskType.EVALUATOR in cluster_spec.jobs:
        eval_thread = threading.Thread(target=_run_single_worker, args=(eval_fn, eval_strategy, cluster_spec, _TaskType.EVALUATOR, 0, session_config), kwargs={'rpc_layer': rpc_layer, 'coord': coord})
        eval_thread.start()
    worker_result = _run_single_worker(worker_fn, strategy, cluster_spec, None, None, session_config, rpc_layer=rpc_layer, coord=coord)
    if eval_thread:
        coord.join([eval_thread])
    return worker_result