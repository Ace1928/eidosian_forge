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
class _KubeCluster(object):
    """A kubernetes cluster.

  Attributes:
    context_name: Kubernetes context name.
    env_vars: Docker env vars.
    shared_docker: Whether the kubernetes cluster shares a docker instance with
      the developer's machine.
  """

    def __init__(self, context_name, shared_docker):
        """Initializes KubeCluster with cluster name.

    Args:
      context_name: Kubernetes context.
      shared_docker: Whether the kubernetes cluster shares a docker instance
        with the developer's machine.
    """
        self.context_name = context_name
        self.shared_docker = shared_docker

    @property
    def env_vars(self):
        return {}