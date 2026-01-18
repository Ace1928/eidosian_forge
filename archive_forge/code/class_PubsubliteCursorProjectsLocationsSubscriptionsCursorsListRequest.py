from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteCursorProjectsLocationsSubscriptionsCursorsListRequest(_messages.Message):
    """A PubsubliteCursorProjectsLocationsSubscriptionsCursorsListRequest
  object.

  Fields:
    pageSize: The maximum number of cursors to return. The service may return
      fewer than this value. If unset or zero, all cursors for the parent will
      be returned.
    pageToken: A page token, received from a previous `ListPartitionCursors`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListPartitionCursors` must match the call
      that provided the page token.
    parent: Required. The subscription for which to retrieve cursors.
      Structured like `projects/{project_number}/locations/{location}/subscrip
      tions/{subscription_id}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)