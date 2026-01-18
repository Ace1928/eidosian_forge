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
def GetOpenIDKeyset(self, jwks_uri=None):
    """Get the JSON Web Key Set for the K8s API server.

    Args:
      jwks_uri: string, the JWKS URI to query for the JSON Web Key Set. If None,
        queries the cluster's built-in endpoint.

    Returns:
      The JSON response as a string.

    Raises:
      Error: If the query failed.
    """
    headers = {'Content-Type': 'application/jwk-set+json'}
    url = None
    try:
        if jwks_uri is not None:
            url = jwks_uri
            return self._WebRequest('GET', url, headers=headers)
        else:
            url = '/openid/v1/jwks'
            return self._ClusterRequest('GET', url, headers=headers)
    except Exception as e:
        raise exceptions.Error('Failed to get JSON Web Key Set from {}: {}'.format(url, e))