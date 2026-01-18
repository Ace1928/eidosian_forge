from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsInstalledAppsListRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsInstalledAppsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of workforce pool installed apps to
      return. If unspecified, at most 50 workforce pool installed apps will be
      returned. The maximum value is 1000; values above 1000 are truncated to
      1000.
    pageToken: Optional. A page token, received from a previous
      `ListWorkforcePoolInstalledApps` call. Provide this to retrieve the
      subsequent page.
    parent: Required. The parent to list installed apps, format:
      'locations/{location}/workforcePools/{workforce_pool}'
    showDeleted: Optional. Whether to return soft-deleted workforce pool
      installed apps.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)