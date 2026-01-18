from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMessagePublishingRoutesPatchRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMessagePublishingRoutesPatchRequest
  object.

  Fields:
    messagePublishingRoute: A MessagePublishingRoute resource to be passed as
      the request body.
    name: Identifier. Name of the MessagePublishRoute resource. It matches
      pattern `projects/*/locations/*/messagePublishingRoutes/`.
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the MessagePublishingRoute resource by the update. The
      fields specified in the update_mask are relative to the resource, not
      the full request. A field will be overwritten if it is in the mask. If
      the user does not provide a mask then all fields will be overwritten.
  """
    messagePublishingRoute = _messages.MessageField('MessagePublishingRoute', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)