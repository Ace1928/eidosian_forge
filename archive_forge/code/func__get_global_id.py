import copy
import six
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
def _get_global_id(cluster_spec, task_type, task_id, chief_task_type):
    """Returns the global id of the given task type in a cluster."""
    if not task_type:
        return 0
    task_type_ordered_list = []
    if chief_task_type in cluster_spec.jobs:
        task_type_ordered_list = [chief_task_type]
    task_type_ordered_list.extend([t for t in sorted(cluster_spec.jobs) if t != chief_task_type and t != PS])
    if PS in cluster_spec.jobs:
        task_type_ordered_list.append(PS)
    next_global_id = 0
    for t in task_type_ordered_list:
        if t == task_type:
            return next_global_id + task_id
        next_global_id += len(cluster_spec.job_tasks(t))
    raise RuntimeError('Internal Error: `task_type` ({}) is not in cluster_spec ({}).'.format(task_type, cluster_spec))