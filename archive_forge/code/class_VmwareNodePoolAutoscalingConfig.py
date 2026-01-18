from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareNodePoolAutoscalingConfig(_messages.Message):
    """NodePoolAutoscaling config for the NodePool to allow for the kubernetes
  to scale NodePool.

  Fields:
    maxReplicas: Maximum number of replicas in the NodePool.
    minReplicas: Minimum number of replicas in the NodePool.
  """
    maxReplicas = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minReplicas = _messages.IntegerField(2, variant=_messages.Variant.INT32)