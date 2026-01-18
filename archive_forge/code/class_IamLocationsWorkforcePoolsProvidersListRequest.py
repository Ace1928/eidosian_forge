from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsProvidersListRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsProvidersListRequest object.

  Fields:
    pageSize: The maximum number of providers to return. If unspecified, at
      most 50 providers are returned. The maximum value is 100; values above
      100 are truncated to 100.
    pageToken: A page token, received from a previous
      `ListWorkforcePoolProviders` call. Provide this to retrieve the
      subsequent page.
    parent: Required. The pool to list providers for. Format:
      `locations/{location}/workforcePools/{workforce_pool_id}`
    showDeleted: Whether to return soft-deleted providers.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)