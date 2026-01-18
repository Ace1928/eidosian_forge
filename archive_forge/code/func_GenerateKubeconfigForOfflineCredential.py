from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import os
import subprocess
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
def GenerateKubeconfigForOfflineCredential(cluster, context, credential_resp):
    """Generates a kubeconfig entry based on offline credential for a Edge Container cluster.

  Args:
    cluster: object, Edge Container cluster.
    context: str, context for the kubeconfig entry.
    credential_resp: Response from GetOfflineCredential API.

  Raises:
      Error: don't have the permission to open kubeconfig file
  """
    kubeconfig = Kubeconfig.Default()
    kubeconfig_for_output = EmptyKubeconfig()
    kubeconfig.contexts[context] = Context(context, context, context)
    kubeconfig_for_output['contexts'].append(Context(context, context, context))
    user_kwargs = {}
    if credential_resp.clientKey is None:
        log.error('Offline credential is missing client key.')
    else:
        user_kwargs['key_data'] = _GetPemDataForKubeconfig(credential_resp.clientKey)
    if credential_resp.clientCertificate is None:
        log.error('Offline credential is missing client certificate.')
    else:
        user_kwargs['cert_data'] = _GetPemDataForKubeconfig(credential_resp.clientCertificate)
    user = User(context, **user_kwargs)
    del user['user']['exec']
    kubeconfig.users[context] = user
    kubeconfig_for_output['users'].append(user)
    port = getattr(cluster, 'port', 443)
    if port is None:
        port = 443
    cluster_kwargs = {}
    if cluster.clusterCaCertificate is None:
        log.warning('Cluster is missing certificate authority data.')
    else:
        cluster_kwargs['ca_data'] = _GetPemDataForKubeconfig(cluster.clusterCaCertificate)
    kubeconfig.clusters[context] = Cluster(context, 'https://{}:{}'.format(cluster.endpoint, port), **cluster_kwargs)
    kubeconfig_for_output['clusters'].append(Cluster(context, 'https://{}:{}'.format(cluster.endpoint, port), **cluster_kwargs))
    kubeconfig.SetCurrentContext(context)
    kubeconfig_for_output['current-context'] = context
    yaml.dump(kubeconfig_for_output, sys.stderr)
    kubeconfig.SaveToFile()
    log.status.Print('A new kubeconfig entry "{}" has been generated and set as the current context.'.format(context))