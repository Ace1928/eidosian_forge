from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdsProjectsLocationsEndpointsListRequest(_messages.Message):
    """A IdsProjectsLocationsEndpointsListRequest object.

  Fields:
    filter: Optional. The filter expression, following the syntax outlined in
      https://google.aip.dev/160.
    orderBy: Optional. One or more fields to compare and use to sort the
      output. See https://google.aip.dev/132#ordering.
    pageSize: Optional. The maximum number of endpoints to return. The service
      may return fewer than this value.
    pageToken: Optional. A page token, received from a previous
      `ListEndpoints` call. Provide this to retrieve the subsequent page. When
      paginating, all other parameters provided to `ListEndpoints` must match
      the call that provided the page token.
    parent: Required. The parent, which owns this collection of endpoints.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)