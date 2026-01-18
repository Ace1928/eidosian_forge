import os
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.experimental import dtensor_strategy_extended
from tensorflow.python.distribute.experimental import dtensor_util
def _parse_dtensor_env_var_from_cluster_resolver(cluster_resolver):
    """Parse the env vars for Dtensor based on the cluster resolver.

  In the multi-client setting, each of the DTensor jobs need to aware of each
  other, and the interface to setup those values are via the envvars. The
  value used by dtensor are different from the existing
  `MultiWorkerMirroredStrategy`. This function will parse the value from
  cluster resolver, and populate the corresponding value for DTensor jobs in the
  `os.environ`.

  Args:
    cluster_resolver: A `tf.distribute.cluster_resolver.ClusterResolver`
      instance.

  Returns:
    A dict of {Str:Str} which contains all the env vars needed by DTensor jobs.
    The value is for verification purpose.

  Raises:
    The value parsed from existing cluster spec is not valid.
  """
    result = {}
    cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_resolver.cluster_spec())
    dtensor_jobs = []
    if 'chief' in cluster_spec.jobs:
        dtensor_jobs.extend(cluster_spec.job_tasks('chief'))
    if 'worker' in cluster_spec.jobs:
        dtensor_jobs.extend(cluster_spec.job_tasks('worker'))
    if None in dtensor_jobs:
        raise ValueError(f'Unexpected dtensor job address from cluster spec: {cluster_spec}')
    result['DTENSOR_JOBS'] = ','.join(dtensor_jobs)
    result['DTENSOR_NUM_CLIENTS'] = str(len(dtensor_jobs))
    if cluster_resolver.task_type == 'chief':
        dtensor_client_id = 0
    elif cluster_resolver.task_type == 'worker':
        dtensor_client_id = cluster_resolver.task_id
        if 'chief' in cluster_spec.jobs:
            dtensor_client_id += 1
    result['DTENSOR_CLIENT_ID'] = str(dtensor_client_id)
    result['DTENSOR_JOB_NAME'] = 'worker'
    return result