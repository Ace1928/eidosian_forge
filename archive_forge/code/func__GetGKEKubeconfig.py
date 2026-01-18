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
def _GetGKEKubeconfig(api_adapter, project, location_id, cluster_id, temp_kubeconfig_dir, internal_ip, cross_connect_subnetwork, private_endpoint_fqdn):
    """The kubeconfig of GKE Cluster is fetched using the GKE APIs.

  The 'KUBECONFIG' value in `os.environ` will be temporarily updated with
  the temporary kubeconfig's path if the kubeconfig arg is not None.
  Consequently, subprocesses started with
  googlecloudsdk.core.execution_utils.Exec will see the temporary KUBECONFIG
  environment variable.

  Using GKE APIs the GKE cluster is validated, and the ClusterConfig object, is
  persisted in the temporarily updated 'KUBECONFIG'.

  Args:
    api_adapter: the GKE api adapter used for running kubernetes commands
    project: string, the project id of the cluster for which kube config is to
      be fetched
    location_id: string, the id of the location to which the cluster belongs
    cluster_id: string, the id of the cluster
    temp_kubeconfig_dir: TemporaryDirectory object
    internal_ip: whether to persist the internal IP of the endpoint.
    cross_connect_subnetwork: full path of the cross connect subnet whose
      endpoint to persist (optional)
    private_endpoint_fqdn: whether to persist the private fqdn.

  Raises:
    Error: If unable to get credentials for kubernetes cluster.

  Returns:
    the path to the kubeconfig file
  """
    kubeconfig = os.path.join(temp_kubeconfig_dir.path, 'kubeconfig')
    old_kubeconfig = encoding.GetEncodedValue(os.environ, 'KUBECONFIG')
    try:
        encoding.SetEncodedValue(os.environ, 'KUBECONFIG', kubeconfig)
        if api_adapter is None:
            api_adapter = gke_api_adapter.NewAPIAdapter('v1')
        cluster_ref = api_adapter.ParseCluster(cluster_id, location_id, project)
        cluster = api_adapter.GetCluster(cluster_ref)
        auth = cluster.masterAuth
        valid_creds = auth and auth.clientCertificate and auth.clientKey
        if not valid_creds and (not c_util.ClusterConfig.UseGCPAuthProvider()):
            raise c_util.Error('Unable to get cluster credentials. User must have edit permission on {}'.format(cluster_ref.projectId))
        c_util.ClusterConfig.Persist(cluster, cluster_ref.projectId, internal_ip, cross_connect_subnetwork, private_endpoint_fqdn)
    finally:
        if old_kubeconfig:
            encoding.SetEncodedValue(os.environ, 'KUBECONFIG', old_kubeconfig)
        else:
            del os.environ['KUBECONFIG']
    return kubeconfig