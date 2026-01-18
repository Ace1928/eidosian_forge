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
def GetOpenIDConfiguration(self, issuer_url=None):
    """Get the OpenID Provider Configuration for the K8s API server.

    Args:
      issuer_url: string, the issuer URL to query for the OpenID Provider
        Configuration. If None, queries the custer's built-in endpoint.

    Returns:
      The JSON response as a string.

    Raises:
      Error: If the query failed.
    """
    headers = {'Content-Type': 'application/json'}
    url = None
    try:
        if issuer_url is not None:
            url = issuer_url.rstrip('/') + '/.well-known/openid-configuration'
            return self._WebRequest('GET', url, headers=headers)
        else:
            url = '/.well-known/openid-configuration'
            return self._ClusterRequest('GET', url, headers=headers)
    except Exception as e:
        raise exceptions.Error('Failed to get OpenID Provider Configuration from {}: {}'.format(url, e))