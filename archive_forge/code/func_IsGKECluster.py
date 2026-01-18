from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
import json
import os
import re
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import format_util
from googlecloudsdk.command_lib.container.fleet.memberships import gke_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from kubernetes import client as kube_client_lib
from kubernetes import config as kube_client_config
from six.moves.urllib.parse import urljoin
def IsGKECluster(kube_client):
    """Returns true if the cluster to be registered is a GKE cluster.

  There is no straightforward way to obtain this information from the cluster
  API server directly. This method uses metadata on the Kubernetes nodes to
  determine the instance ID. The instance ID field is unique to GKE clusters:
  Kubernetes-on-GCE clusters do not have this field. This test doesn't work in
  identifing a GKE cluster with zero nodes.

  Args:
    kube_client: A Kubernetes client for the cluster to be registered.

  Raises:
      exceptions.Error: if failing there's a permission error or for invalid
      command.

  Returns:
    bool: True if kubeclient communicates with a GKE Cluster, false otherwise.
  """
    if kube_client.processor and kube_client.processor.gke_cluster_self_link:
        return True
    vm_instance_id, err = kube_client.GetResourceField(None, 'nodes', '.items[*].metadata.annotations.container\\.googleapis\\.com/instance_id')
    if err:
        raise exceptions.Error('kubectl returned non-zero status code: {}'.format(err))
    if not vm_instance_id:
        return False
    return True