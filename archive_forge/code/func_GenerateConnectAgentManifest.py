from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import gkehub_api_util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.container.fleet import base as hub_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
def GenerateConnectAgentManifest(membership_ref, image_pull_secret_content=None, is_upgrade=None, namespace=None, proxy=None, registry=None, version=None, release_track=None):
    """Generated the Connect Agent to apply to the registered cluster.

  Args:
    membership_ref: the full resource name of the membership.
    image_pull_secret_content: The image pull secret content to use for private
      registries or None if it is not available.
    is_upgrade: Is this is an upgrade operation, or None if it is not available.
    namespace: The namespace of the Connect Agent, or None if it is not
      available.
    proxy: The proxy address or None if it is not available.
    registry: The registry to pull the Connect Agent image if not using
      gcr.io/gkeconnect, or None if it is not available.
    version: The version of the Connect Agent to install/upgrade, or None if it
      is not available.
    release_track: the release_track used in the gcloud command, or None if it
      is not available.

  Returns:
    the GenerateConnectManifest from API.

  Raises:
    apitools.base.py.HttpError: if the request returns an HTTP error.
  """
    client = gkehub_api_util.GetApiClientForTrack(release_track)
    messages = client.MESSAGES_MODULE
    request = messages.GkehubProjectsLocationsMembershipsGenerateConnectManifestRequest(name=membership_ref)
    if image_pull_secret_content:
        request.imagePullSecretContent = image_pull_secret_content.encode('utf-8')
    if is_upgrade:
        request.isUpgrade = is_upgrade
    if namespace:
        request.namespace = namespace
    if proxy:
        request.proxy = proxy.encode('ascii')
    if registry:
        request.registry = registry
    if version:
        request.version = version
    return client.projects_locations_memberships.GenerateConnectManifest(request)