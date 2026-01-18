from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import os
import subprocess
from googlecloudsdk.api_lib.container import kubeconfig as kubeconfig_util
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.command_lib.container.fleet import gateway
from googlecloudsdk.command_lib.container.fleet import gwkubeconfig_util
from googlecloudsdk.command_lib.container.gkemulticloud import errors
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
def GenerateContext(kind, project_id, location, cluster_id):
    """Generates a kubeconfig context for an Anthos Multi-Cloud cluster.

  Args:
    kind: str, kind of the cluster e.g. aws, azure.
    project_id: str, project ID accociated with the cluster.
    location: str, Google location of the cluster.
    cluster_id: str, ID of the cluster.

  Returns:
    The context for the kubeconfig entry.
  """
    template = 'gke_{kind}_{project_id}_{location}_{cluster_id}'
    return template.format(kind=kind, project_id=project_id, location=location, cluster_id=cluster_id)