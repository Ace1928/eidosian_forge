from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.training import server_lib
def _validate_cluster_spec(cluster_spec, task_type, task_id):
    """Validates `cluster_spec`.

  It checks:
  1) task type is one of "chief", "worker", "ps", "evaluator", or not provided
     (None).
  2) whether there is such a task type as `task_type` in the `cluster_spec`. The
     only exception is `evaluator`. In other words, it is still a valid
     configuration when `task_type` is `evaluator` but it doesn't appear in
     `cluster_spec`. This is to be compatible with `TF_CONFIG` in Estimator.
  3) whether there is at most one "chief" job.
  4) whether there is at most one "evaluator" job.
  5) whether the `task_id` is smaller than the number of tasks for that
     particular `task_type`.

  Args:
    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object to be validated.
    task_type: string indicating the type of the task.
    task_id: the id of the `task_type` in this cluster.

  Raises:
    ValueError: if `cluster_spec` fails any check.
  """
    allowed_task_types = ('chief', 'worker', 'evaluator', 'ps', None)
    cluster_spec = normalize_cluster_spec(cluster_spec)
    if any((job not in allowed_task_types for job in cluster_spec.jobs)):
        raise ValueError('Disallowed task type found in cluster spec. Allowed types are {} and the cluster spec is {}.'.format(allowed_task_types, cluster_spec))
    if task_type not in allowed_task_types:
        raise ValueError('Unrecognized task_type: {}, valid task types are: {}'.format(task_type, allowed_task_types))
    if task_type and task_type not in cluster_spec.jobs and (task_type != 'evaluator'):
        raise ValueError('`task_type` %r not found in cluster_spec.' % task_type)
    if task_count(cluster_spec, 'chief') > 1:
        raise ValueError("There must be at most one 'chief' job.")
    if task_count(cluster_spec, 'evaluator') > 1:
        raise ValueError("There must be at most one 'evaluator' job.")
    if task_type in cluster_spec.jobs and task_id >= task_count(cluster_spec, task_type):
        raise ValueError('The `task_id` %d exceeds the maximum id of %s.' % (task_id, task_type))