from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
import subprocess
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from six.moves import range
def RunDistributed(module_name, package_root, num_ps, num_workers, num_evaluators, start_port, user_args=None):
    """Create a cluster configuration and start processes for the cluster.

  Args:
    module_name: str. Python module to use as the task.
    package_root: str. Absolute path to the package root of the module.
    num_ps: int. Number of parameter servers
    num_workers: int. Number of workers.
    num_evaluators: int. Number of evaluators.
    start_port: int. First port for the contiguous block of ports used
      by the cluster.
    user_args: [str]. Additional user args for the task. Any relative paths will
      not work.
  Returns:
    int. the retval of primary subprocess
  """
    ports = list(range(start_port, start_port + num_ps + num_workers + 1))
    cluster = {GetPrimaryNodeName(): ['localhost:{port}'.format(port=ports[0])], 'ps': ['localhost:{port}'.format(port=p) for p in ports[1:num_ps + 1]], 'worker': ['localhost:{port}'.format(port=p) for p in ports[num_ps + 1:]]}
    for task_type, addresses in cluster.items():
        if task_type != GetPrimaryNodeName():
            for i in range(len(addresses)):
                MakeProcess(module_name, package_root, args=user_args, task_type=task_type, index=i, cluster=cluster)
    for i in range(num_evaluators):
        MakeProcess(module_name, package_root, args=user_args, task_type='evaluator', index=i, cluster=cluster)
    return MakeProcess(module_name, package_root, args=user_args, task_type=GetPrimaryNodeName(), index=0, cluster=cluster)