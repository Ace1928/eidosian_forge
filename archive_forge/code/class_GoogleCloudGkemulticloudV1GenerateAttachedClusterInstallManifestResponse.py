from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1GenerateAttachedClusterInstallManifestResponse(_messages.Message):
    """Response message for
  `AttachedClusters.GenerateAttachedClusterInstallManifest` method.

  Fields:
    manifest: A set of Kubernetes resources (in YAML format) to be applied to
      the cluster to be attached.
  """
    manifest = _messages.StringField(1)