from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest(_messages.Message):
    """A
  GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest
  object.

  Fields:
    attachedClusterId: Required. A client provided ID of the resource. Must be
      unique within the parent resource. The provided ID will be part of the
      AttachedCluster resource name formatted as
      `projects//locations//attachedClusters/`. Valid characters are `/a-z-/`.
      Cannot be longer than 63 characters. When generating an install manifest
      for importing an existing Membership resource, the attached_cluster_id
      field must be the Membership id. Membership names are formatted as
      `projects//locations//memberships/`.
    parent: Required. The parent location where this AttachedCluster resource
      will be created. Location names are formatted as `projects//locations/`.
      See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
    platformVersion: Required. The platform version for the cluster (e.g.
      `1.19.0-gke.1000`). You can list all supported versions on a given
      Google Cloud region by calling GetAttachedServerConfig.
    proxyConfig_kubernetesSecret_name: Name of the kubernetes secret.
    proxyConfig_kubernetesSecret_namespace: Namespace in which the kubernetes
      secret is stored.
  """
    attachedClusterId = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)
    platformVersion = _messages.StringField(3)
    proxyConfig_kubernetesSecret_name = _messages.StringField(4)
    proxyConfig_kubernetesSecret_namespace = _messages.StringField(5)