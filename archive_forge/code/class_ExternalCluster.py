from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import subprocess
import sys
from googlecloudsdk.command_lib.code import run_subprocess
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import times
import six
class ExternalCluster(_KubeCluster):
    """A external kubernetes cluster.

  Attributes:
    context_name: Kubernetes context name.
    env_vars: Docker environment variables.
    shared_docker: Whether the kubernetes cluster shares a docker instance with
      the developer's machine.
  """

    def __init__(self, cluster_name):
        """Initializes ExternalCluster with profile name.

    Args:
      cluster_name: Name of the cluster.
    """
        super(ExternalCluster, self).__init__(cluster_name, False)