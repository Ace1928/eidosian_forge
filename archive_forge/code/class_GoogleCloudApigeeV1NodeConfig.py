from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1NodeConfig(_messages.Message):
    """NodeConfig for setting the min/max number of nodes associated with the
  environment.

  Fields:
    currentAggregateNodeCount: Output only. The current total number of
      gateway nodes that each environment currently has across all instances.
    maxNodeCount: Optional. The maximum total number of gateway nodes that the
      is reserved for all instances that has the specified environment. If not
      specified, the default is determined by the recommended maximum number
      of nodes for that gateway.
    minNodeCount: Optional. The minimum total number of gateway nodes that the
      is reserved for all instances that has the specified environment. If not
      specified, the default is determined by the recommended minimum number
      of nodes for that gateway.
  """
    currentAggregateNodeCount = _messages.IntegerField(1)
    maxNodeCount = _messages.IntegerField(2)
    minNodeCount = _messages.IntegerField(3)