from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMessagePublishingRoutesGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMessagePublishingRoutesGetRequest
  object.

  Fields:
    name: Required. A name of the MessagePublishingRoute to get. Must be in
      the format `projects/*/locations/*/messagePublishingRoutes/*`.
  """
    name = _messages.StringField(1, required=True)