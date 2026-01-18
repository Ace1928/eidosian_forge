from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeGroupsAddNodesRequest(_messages.Message):
    """A NodeGroupsAddNodesRequest object.

  Fields:
    additionalNodeCount: Count of additional nodes to be added to the node
      group.
  """
    additionalNodeCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)