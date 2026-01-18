from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheOriginsCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEdgeCacheOriginsCreateRequest object.

  Fields:
    edgeCacheOrigin: A EdgeCacheOrigin resource to be passed as the request
      body.
    edgeCacheOriginId: Required. The short name of the EdgeCacheOrigin
      resource to create, such as `MyEdgeCacheOrigin`.
    parent: Required. The parent resource of the EdgeCacheOrigin resource.
      Must be in the format `projects/*/locations/global`.
  """
    edgeCacheOrigin = _messages.MessageField('EdgeCacheOrigin', 1)
    edgeCacheOriginId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)