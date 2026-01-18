from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheOriginsGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEdgeCacheOriginsGetRequest object.

  Fields:
    name: Required. The name of the EdgeCacheOrigin resource to get. Must be
      in the format `projects/*/locations/global/edgeCacheOrigins/*`.
  """
    name = _messages.StringField(1, required=True)