from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureOnlineStoreBigtableAutoScaling(_messages.Message):
    """A GoogleCloudAiplatformV1FeatureOnlineStoreBigtableAutoScaling object.

  Fields:
    cpuUtilizationTarget: Optional. A percentage of the cluster's CPU
      capacity. Can be from 10% to 80%. When a cluster's CPU utilization
      exceeds the target that you have set, Bigtable immediately adds nodes to
      the cluster. When CPU utilization is substantially lower than the
      target, Bigtable removes nodes. If not set will default to 50%.
    maxNodeCount: Required. The maximum number of nodes to scale up to. Must
      be greater than or equal to min_node_count, and less than or equal to 10
      times of 'min_node_count'.
    minNodeCount: Required. The minimum number of nodes to scale down to. Must
      be greater than or equal to 1.
  """
    cpuUtilizationTarget = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxNodeCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    minNodeCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)