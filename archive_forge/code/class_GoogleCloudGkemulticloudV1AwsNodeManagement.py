from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsNodeManagement(_messages.Message):
    """AwsNodeManagement defines the set of node management features turned on
  for an AWS node pool.

  Fields:
    autoRepair: Optional. Whether or not the nodes will be automatically
      repaired. When set to true, the nodes in this node pool will be
      monitored and if they fail health checks consistently over a period of
      time, an automatic repair action will be triggered to replace them with
      new nodes.
  """
    autoRepair = _messages.BooleanField(1)