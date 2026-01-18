from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1SurgeSettings(_messages.Message):
    """SurgeSettings contains the parameters for Surge update.

  Fields:
    maxSurge: Optional. The maximum number of nodes that can be created beyond
      the current size of the node pool during the update process.
    maxUnavailable: Optional. The maximum number of nodes that can be
      simultaneously unavailable during the update process. A node is
      considered unavailable if its status is not Ready.
  """
    maxSurge = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxUnavailable = _messages.IntegerField(2, variant=_messages.Variant.INT32)