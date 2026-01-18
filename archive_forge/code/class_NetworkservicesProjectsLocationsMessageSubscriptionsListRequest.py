from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMessageSubscriptionsListRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMessageSubscriptionsListRequest
  object.

  Fields:
    pageSize: Optional. Maximum number of MessageSubscriptions to return per
      call.
    pageToken: Optional. The value returned by the last
      `ListMessageSubscriptionsResponse` Indicates that this is a continuation
      of a prior `ListMessageSubscriptions` call, and that the system should
      return the next page of data.
    parent: Required. The project and location from which the
      MessageSubscriptions should be listed, specified in the format
      `projects/*/locations/*`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)