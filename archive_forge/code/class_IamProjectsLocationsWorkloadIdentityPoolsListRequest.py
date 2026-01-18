from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsListRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsListRequest object.

  Fields:
    pageSize: The maximum number of pools to return. If unspecified, at most
      50 pools are returned. The maximum value is 1000; values above are 1000
      truncated to 1000.
    pageToken: A page token, received from a previous
      `ListWorkloadIdentityPools` call. Provide this to retrieve the
      subsequent page.
    parent: Required. The parent resource to list pools for.
    showDeleted: Whether to return soft-deleted pools.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)