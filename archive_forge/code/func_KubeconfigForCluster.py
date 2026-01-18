from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import fnmatch
import io
import json
import os
import re
import signal
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def KubeconfigForCluster(project, region, cluster):
    """Get the kubeconfig of a GKE cluster.

  If the kubeconfig for the GKE cluster already exists locally, use it;
  Otherwise run a gcloud command to get the credential for it.

  Args:
    project: The project ID of the cluster.
    region: The region of the cluster.
    cluster: The name of the cluster.

  Returns:
    None

  Raises:
    Error: The error occured when it failed to get credential for the cluster.
  """
    context = 'gke_{project}_{region}_{cluster}'.format(project=project, region=region, cluster=cluster)
    command = ['config', 'use-context', context]
    _, err = RunKubectl(command)
    if err is None:
        return None
    args = ['container', 'clusters', 'get-credentials', cluster, '--region', region, '--project', project]
    _, err = _RunGcloud(args)
    if err:
        raise exceptions.ConfigSyncError('Error getting credential for cluster {}: {}'.format(cluster, err))