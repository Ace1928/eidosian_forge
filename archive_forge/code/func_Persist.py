from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
@classmethod
def Persist(cls, cluster, project_id, use_internal_ip=False, cross_connect_subnetwork=None, use_private_fqdn=None, use_dns_endpoint=None):
    """Saves config data for the given cluster.

    Persists config file and kubernetes auth file for the given cluster
    to cloud-sdk config directory and returns ClusterConfig object
    encapsulating the same data.

    Args:
      cluster: valid Cluster message to persist config data for.
      project_id: project that owns this cluster.
      use_internal_ip: whether to persist the internal IP of the endpoint.
      cross_connect_subnetwork: full path of the cross connect subnet whose
        endpoint to persist (optional)
      use_private_fqdn: whether to persist the private fqdn.
      use_dns_endpoint: whether to generate dns endpoint address.

    Returns:
      ClusterConfig of the persisted data.
    Raises:
      Error: if cluster has no endpoint (will be the case for first few
        seconds while cluster is PROVISIONING).
    """
    endpoint = _GetClusterEndpoint(cluster, use_internal_ip, cross_connect_subnetwork, use_private_fqdn, use_dns_endpoint)
    kwargs = {'cluster_name': cluster.name, 'zone_id': cluster.zone, 'project_id': project_id, 'server': 'https://' + endpoint}
    if use_dns_endpoint:
        kwargs['dns_endpoint'] = endpoint
    auth = cluster.masterAuth
    if auth and auth.clusterCaCertificate:
        kwargs['ca_data'] = auth.clusterCaCertificate
    else:
        log.warning('Cluster is missing certificate authority data.')
    if cls.UseGCPAuthProvider():
        kwargs['auth_provider'] = 'gcp'
    elif auth.clientCertificate and auth.clientKey:
        kwargs['client_key_data'] = auth.clientKey
        kwargs['client_cert_data'] = auth.clientCertificate
    c_config = cls(**kwargs)
    c_config.GenKubeconfig()
    return c_config