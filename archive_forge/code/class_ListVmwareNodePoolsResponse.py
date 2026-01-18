from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListVmwareNodePoolsResponse(_messages.Message):
    """Response message for listing VMware node pools.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    unreachable: Locations that could not be reached.
    vmwareNodePools: The node pools from the specified parent resource.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    vmwareNodePools = _messages.MessageField('VmwareNodePool', 3, repeated=True)