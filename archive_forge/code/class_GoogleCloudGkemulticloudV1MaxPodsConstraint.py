from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1MaxPodsConstraint(_messages.Message):
    """Constraints applied to pods.

  Fields:
    maxPodsPerNode: Required. The maximum number of pods to schedule on a
      single node.
  """
    maxPodsPerNode = _messages.IntegerField(1)