from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkforcePoolsResponse(_messages.Message):
    """Response message for ListWorkforcePools.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    workforcePools: A list of pools.
  """
    nextPageToken = _messages.StringField(1)
    workforcePools = _messages.MessageField('WorkforcePool', 2, repeated=True)