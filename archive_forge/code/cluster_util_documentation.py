from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.container.gkemulticloud import errors
Verifies the install agent is deployed and running on the target cluster.

  Accesses the cluster and checks if the install agent is running to ensure
  subsequent operations can succeed. Raises MissingAttachedInstallAgent if the
  agent is not running or it can't be determined.

  Args:
    kube_client: Client to the cluster.
  