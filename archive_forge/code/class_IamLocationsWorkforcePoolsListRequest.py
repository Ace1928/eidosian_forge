from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsListRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsListRequest object.

  Fields:
    location: The location of the pool. Format: `locations/{location}`.
    pageSize: The maximum number of pools to return. If unspecified, at most
      50 pools will be returned. The maximum value is 1000; values above 1000
      are truncated to 1000.
    pageToken: A page token, received from a previous `ListWorkforcePools`
      call. Provide this to retrieve the subsequent page.
    parent: Required. The parent resource to list pools for. Format:
      `organizations/{org-id}`.
    showDeleted: Whether to return soft-deleted pools.
  """
    location = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4)
    showDeleted = _messages.BooleanField(5)