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
def CheckClusterHasNodePools(cluster_client, cluster_ref):
    """Checks and gives a warning if the cluster does not have a node pool."""
    try:
        if not cluster_client.HasNodePools(cluster_ref):
            log.warning('Cluster does not have a node pool. To use Connect Gateway, ensure you have at least one Linux node pool running.')
    except:
        pass