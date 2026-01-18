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
class KubeconfigProcessor(object):
    """A helper class that processes kubeconfig and context arguments."""

    def __init__(self, api_adapter, gke_uri, gke_cluster, kubeconfig, internal_ip, cross_connect_subnetwork, private_endpoint_fqdn, context):
        """Constructor for KubeconfigProcessor.

    Args:
      api_adapter: the GKE api adapter used for running kubernetes commands
      gke_uri: the URI of the GKE cluster; for example,
        'https://container.googleapis.com/v1/projects/my-project/locations/us-central1-a/clusters/my-cluster'
      gke_cluster: the "location/name" of the GKE cluster. The location can be a
        zone or a region for e.g `us-central1-a/my-cluster`
      kubeconfig: the kubeconfig path
      internal_ip: whether to persist the internal IP of the endpoint.
      cross_connect_subnetwork: full path of the cross connect subnet whose
        endpoint to persist (optional)
      private_endpoint_fqdn: whether to persist the private fqdn.
      context: the context to use

    Raises:
      exceptions.Error: if kubectl is not installed
    """
        self.api_adapter = api_adapter
        self.gke_uri = gke_uri
        self.gke_cluster = gke_cluster
        self.kubeconfig = kubeconfig
        self.internal_ip = internal_ip
        self.cross_connect_subnetwork = cross_connect_subnetwork
        self.private_endpoint_fqdn = private_endpoint_fqdn
        self.context = context
        if not c_util.CheckKubectlInstalled():
            raise exceptions.Error('kubectl not installed.')
        self.gke_cluster_self_link = None
        self.gke_cluster_uri = None

    def GetKubeconfigAndContext(self, temp_kubeconfig_dir):
        """Gets the kubeconfig, cluster context and resource link from arguments and defaults.

    Args:
      temp_kubeconfig_dir: a TemporaryDirectoryObject.

    Returns:
      the kubeconfig filepath and context name

    Raises:
      calliope_exceptions.MinimumArgumentException: if a kubeconfig file cannot
        be deduced from the command line flags or environment
      exceptions.Error: if the context does not exist in the deduced kubeconfig
        file
    """
        if self.gke_uri or self.gke_cluster:
            cluster_project = None
            if self.gke_uri:
                cluster_project, location, name = gke_util.ParseGKEURI(self.gke_uri)
            else:
                cluster_project = properties.VALUES.core.project.GetOrFail()
                location, name = gke_util.ParseGKECluster(self.gke_cluster)
            self.gke_cluster_self_link, self.gke_cluster_uri = gke_util.ConstructGKEClusterResourceLinkAndURI(cluster_project, location, name)
            return (_GetGKEKubeconfig(self.api_adapter, cluster_project, location, name, temp_kubeconfig_dir, self.internal_ip, self.cross_connect_subnetwork, self.private_endpoint_fqdn), None)
        if not self.kubeconfig and encoding.GetEncodedValue(os.environ, 'KUBERNETES_SERVICE_PORT') and encoding.GetEncodedValue(os.environ, 'KUBERNETES_SERVICE_HOST'):
            return (None, None)
        kubeconfig_file = self.kubeconfig or encoding.GetEncodedValue(os.environ, 'KUBECONFIG') or '~/.kube/config'
        kubeconfig = files.ExpandHomeDir(kubeconfig_file)
        if not kubeconfig:
            raise calliope_exceptions.MinimumArgumentException(['--kubeconfig'], 'Please specify --kubeconfig, set the $KUBECONFIG environment variable, or ensure that $HOME/.kube/config exists')
        kc = kconfig.Kubeconfig.LoadFromFile(kubeconfig)
        context_name = self.context
        if context_name not in kc.contexts:
            raise exceptions.Error('context [{}] does not exist in kubeconfig [{}]'.format(context_name, kubeconfig))
        return (kubeconfig, context_name)

    def GetKubeClient(self, kubeconfig=None, context=None):
        """Gets a client derived from the kubeconfig and context.

    Args:
      kubeconfig: path to a kubeconfig file, None if in-cluster config.
      context: the kubeconfig context to use, None if in-cluster config.

    Returns:
      kubernetes.client.ApiClient
    """
        if kubeconfig is not None:
            kube_client_config.load_kube_config(config_file=kubeconfig, context=context)
            return kube_client_lib.ApiClient()
        else:
            kube_client_config.load_incluster_config()
            return kube_client_lib.ApiClient()