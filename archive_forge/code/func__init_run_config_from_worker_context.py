import copy
import six
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
def _init_run_config_from_worker_context(config, worker_context):
    """Initializes run config from distribute coordinator's worker context."""
    config._service = None
    config._cluster_spec = worker_context.cluster_spec
    config._task_type = worker_context.task_type
    config._task_id = worker_context.task_id
    config._evaluation_master = worker_context.master_target
    config._master = worker_context.master_target
    config._is_chief = worker_context.is_chief
    if config._cluster_spec:
        if config._task_type != EVALUATOR:
            config._num_ps_replicas = _count_ps(config._cluster_spec)
            config._num_worker_replicas = _count_worker(config._cluster_spec, chief_task_type=CHIEF)
            config._global_id_in_cluster = _get_global_id(config._cluster_spec, config._task_type, config._task_id, chief_task_type=CHIEF)
        else:
            config._cluster_spec = server_lib.ClusterSpec({})
            config._num_ps_replicas = 0
            config._num_worker_replicas = 0
            config._global_id_in_cluster = None
    else:
        config._global_id_in_cluster = 0
        config._num_ps_replicas = 0
        config._num_worker_replicas = 1