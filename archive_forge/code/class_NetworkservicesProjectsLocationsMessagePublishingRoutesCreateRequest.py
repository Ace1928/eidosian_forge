from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMessagePublishingRoutesCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMessagePublishingRoutesCreateRequest
  object.

  Fields:
    messagePublishingRoute: A MessagePublishingRoute resource to be passed as
      the request body.
    messagePublishingRouteId: Required. Short name of the
      MessagePublishingRoute resource to be created.
    parent: Required. The parent resource of the MessagePublishingRoute. Must
      be in the format `projects/*/locations/*`.
  """
    messagePublishingRoute = _messages.MessageField('MessagePublishingRoute', 1)
    messagePublishingRouteId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)