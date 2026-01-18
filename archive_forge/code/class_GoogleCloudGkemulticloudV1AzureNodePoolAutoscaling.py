from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureNodePoolAutoscaling(_messages.Message):
    """Configuration related to Kubernetes cluster autoscaler. The Kubernetes
  cluster autoscaler will automatically adjust the size of the node pool based
  on the cluster load.

  Fields:
    maxNodeCount: Required. Maximum number of nodes in the node pool. Must be
      greater than or equal to min_node_count and less than or equal to 50.
    minNodeCount: Required. Minimum number of nodes in the node pool. Must be
      greater than or equal to 1 and less than or equal to max_node_count.
  """
    maxNodeCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minNodeCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)