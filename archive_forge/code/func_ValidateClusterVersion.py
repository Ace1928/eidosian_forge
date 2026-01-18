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
def ValidateClusterVersion(cluster, cluster_id):
    """Validates the cluster version.

  Args:
    cluster: object, Anthos Multi-cloud cluster.
    cluster_id: str, the cluster ID.

  Raises:
      UnsupportedClusterVersion: cluster version is not supported.
      MissingClusterField: expected cluster field is missing.
  """
    version = _GetSemver(cluster, cluster_id)
    if version < semver.SemVer('1.20.0'):
        raise errors.UnsupportedClusterVersion('The command get-credentials is supported in cluster version 1.20 and newer. For older versions, use get-kubeconfig.')