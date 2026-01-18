from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ResourcePoolAutoscalingSpec(_messages.Message):
    """The min/max number of replicas allowed if enabling autoscaling

  Fields:
    maxReplicaCount: Optional. max replicas in the node pool, must be \\u2265
      replica_count and > min_replica_count or will throw error
    minReplicaCount: Optional. min replicas in the node pool, must be \\u2264
      replica_count and < max_replica_count or will throw error
  """
    maxReplicaCount = _messages.IntegerField(1)
    minReplicaCount = _messages.IntegerField(2)