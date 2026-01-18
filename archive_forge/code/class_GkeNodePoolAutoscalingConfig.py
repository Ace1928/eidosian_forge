from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeNodePoolAutoscalingConfig(_messages.Message):
    """GkeNodePoolAutoscaling contains information the cluster autoscaler needs
  to adjust the size of the node pool to the current cluster usage.

  Fields:
    maxNodeCount: The maximum number of nodes in the node pool. Must be >=
      min_node_count, and must be > 0. Note: Quota must be sufficient to scale
      up the cluster.
    minNodeCount: The minimum number of nodes in the node pool. Must be >= 0
      and <= max_node_count.
  """
    maxNodeCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minNodeCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)