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
def _GetConnectGatewayEndpoint():
    """Gets the corresponding Connect Gateway endpoint for Multicloud environment.

  http://g3doc/cloud/kubernetes/multicloud/g3doc/oneplatform/team/how-to/hub

  Returns:
    The Connect Gateway endpoint.

  Raises:
    Error: Unknown API override.
  """
    endpoint = properties.VALUES.api_endpoint_overrides.gkemulticloud.Get()
    if endpoint is None or endpoint.endswith('gkemulticloud.googleapis.com/') or endpoint.endswith('preprod-gkemulticloud.sandbox.googleapis.com/'):
        return 'connectgateway.googleapis.com'
    if 'staging-gkemulticloud' in endpoint:
        return 'staging-connectgateway.sandbox.googleapis.com'
    if endpoint.startswith('http://localhost') or endpoint.endswith('gkemulticloud.sandbox.googleapis.com/'):
        return 'autopush-connectgateway.sandbox.googleapis.com'
    raise errors.UnknownApiEndpointOverrideError('gkemulticloud')