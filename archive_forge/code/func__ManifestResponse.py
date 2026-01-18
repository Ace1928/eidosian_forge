from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import resources as cloud_resources
import six
def _ManifestResponse(self, client, messages, option):
    return getattr(client.projects_locations_memberships.GenerateConnectManifest(messages.GkehubProjectsLocationsMembershipsGenerateConnectManifestRequest(imagePullSecretContent=six.ensure_binary(option.image_pull_secret_content), isUpgrade=option.is_upgrade, name=option.membership_ref, connectAgent_namespace=option.namespace, connectAgent_proxy=six.ensure_binary(option.proxy), registry=option.registry, version=option.version)), 'manifest')