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
def _ConnectGatewayKubeconfig(kubeconfig, cluster, cluster_id, context, cmd_path):
    """Generates the Connect Gateway kubeconfig entry.

  Args:
    kubeconfig: object, Kubeconfig object.
    cluster: object, Anthos Multi-cloud cluster.
    cluster_id: str, the cluster ID.
    context: str, context for the kubeconfig entry.
    cmd_path: str, authentication provider command path.

  Raises:
      errors.MissingClusterField: cluster is missing required fields.
  """
    if cluster.fleet is None or cluster.fleet.membership is None:
        raise errors.MissingClusterField(cluster_id, 'Fleet membership', STILL_PROVISIONING_MSG)
    server = 'https://{}/v1/{}'.format(_GetConnectGatewayEndpoint(), cluster.fleet.membership)
    user_kwargs = {'auth_provider': 'gcp', 'auth_provider_cmd_path': cmd_path}
    kubeconfig.users[context] = kubeconfig_util.User(context, **user_kwargs)
    kubeconfig.clusters[context] = gwkubeconfig_util.Cluster(context, server)