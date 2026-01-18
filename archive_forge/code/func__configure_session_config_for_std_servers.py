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
def _configure_session_config_for_std_servers(strategy, eval_strategy, session_config, cluster_spec, task_type, task_id):
    """Call strategy's `configure` to mutate the session_config.

  The session_config is currently needed as default config for a TensorFlow
  server. In the future, we should be able to remove this method and only pass
  the session config to a client session.
  """
    if task_type == _TaskType.EVALUATOR:
        if eval_strategy:
            eval_strategy.configure(session_config=session_config)
    else:
        strategy = copy.deepcopy(strategy)
        strategy.configure(session_config=session_config, cluster_spec=cluster_spec, task_type=task_type, task_id=task_id)
    del session_config.device_filters[:]