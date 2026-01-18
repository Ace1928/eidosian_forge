from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodePoolAutoscaling(_messages.Message):
    """NodePoolAutoscaling contains information required by cluster autoscaler
  to adjust the size of the node pool to the current cluster usage.

  Enums:
    LocationPolicyValueValuesEnum: Location policy used when scaling up a
      nodepool.

  Fields:
    autoprovisioned: Can this node pool be deleted automatically.
    enabled: Is autoscaling enabled for this node pool.
    locationPolicy: Location policy used when scaling up a nodepool.
    maxNodeCount: Maximum number of nodes for one location in the NodePool.
      Must be >= min_node_count. There has to be enough quota to scale up the
      cluster.
    minNodeCount: Minimum number of nodes for one location in the NodePool.
      Must be >= 1 and <= max_node_count.
    totalMaxNodeCount: Maximum number of nodes in the node pool. Must be
      greater than total_min_node_count. There has to be enough quota to scale
      up the cluster. The total_*_node_count fields are mutually exclusive
      with the *_node_count fields.
    totalMinNodeCount: Minimum number of nodes in the node pool. Must be
      greater than 1 less than total_max_node_count. The total_*_node_count
      fields are mutually exclusive with the *_node_count fields.
  """

    class LocationPolicyValueValuesEnum(_messages.Enum):
        """Location policy used when scaling up a nodepool.

    Values:
      LOCATION_POLICY_UNSPECIFIED: Not set.
      BALANCED: BALANCED is a best effort policy that aims to balance the
        sizes of different zones.
      ANY: ANY policy picks zones that have the highest capacity available.
    """
        LOCATION_POLICY_UNSPECIFIED = 0
        BALANCED = 1
        ANY = 2
    autoprovisioned = _messages.BooleanField(1)
    enabled = _messages.BooleanField(2)
    locationPolicy = _messages.EnumField('LocationPolicyValueValuesEnum', 3)
    maxNodeCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    minNodeCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    totalMaxNodeCount = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    totalMinNodeCount = _messages.IntegerField(7, variant=_messages.Variant.INT32)