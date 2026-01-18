import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def _resolve_task_configuration(self):
    """Creates a mapping of hostnames to the number of tasks allocated on it.

    Reads the SLURM environment to determine the nodes involved in the current
    job step and number of tasks running on each node.

    Returns a dictionary mapping each hostname to the number of tasks.
    """
    hostlist = self._resolve_hostlist()
    tasks_per_node = expand_tasks_per_node(_get_slurm_var('STEP_TASKS_PER_NODE'))
    return {host: num_tasks for host, num_tasks in zip(hostlist, tasks_per_node)}