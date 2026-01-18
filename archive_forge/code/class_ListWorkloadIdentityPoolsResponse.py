from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkloadIdentityPoolsResponse(_messages.Message):
    """Response message for ListWorkloadIdentityPools.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    workloadIdentityPools: A list of pools.
  """
    nextPageToken = _messages.StringField(1)
    workloadIdentityPools = _messages.MessageField('WorkloadIdentityPool', 2, repeated=True)