from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureClusterResources(_messages.Message):
    """Managed Azure resources for the cluster. The values could change and be
  empty, depending on the state of the cluster.

  Fields:
    controlPlaneApplicationSecurityGroupId: Output only. The ARM ID of the
      control plane application security group.
    networkSecurityGroupId: Output only. The ARM ID of the cluster network
      security group.
  """
    controlPlaneApplicationSecurityGroupId = _messages.StringField(1)
    networkSecurityGroupId = _messages.StringField(2)