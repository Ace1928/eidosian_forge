from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsFleetsListRequest(_messages.Message):
    """A GkehubProjectsLocationsFleetsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of fleets to return. The service
      may return fewer than this value. If unspecified, at most 200 fleets
      will be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: Optional. A page token, received from a previous `ListFleets`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListFleets` must match the call that
      provided the page token.
    parent: Required. The organization or project to list for Fleets under, in
      the format `organizations/*/locations/*` or `projects/*/locations/*`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)