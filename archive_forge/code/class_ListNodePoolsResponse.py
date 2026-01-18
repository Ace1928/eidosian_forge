from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNodePoolsResponse(_messages.Message):
    """ListNodePoolsResponse is the result of ListNodePoolsRequest.

  Fields:
    nodePools: A list of node pools for a cluster.
  """
    nodePools = _messages.MessageField('NodePool', 1, repeated=True)