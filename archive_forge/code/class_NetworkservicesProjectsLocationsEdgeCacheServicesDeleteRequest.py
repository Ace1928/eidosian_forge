from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheServicesDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEdgeCacheServicesDeleteRequest object.

  Fields:
    name: Required. The name of the EdgeCacheService resource to delete. Must
      be in the format `projects/*/locations/global/edgeCacheServices/*`.
  """
    name = _messages.StringField(1, required=True)