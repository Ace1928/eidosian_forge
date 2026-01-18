from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheServicesCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEdgeCacheServicesCreateRequest object.

  Fields:
    edgeCacheService: A EdgeCacheService resource to be passed as the request
      body.
    edgeCacheServiceId: Required. The short name of the EdgeCacheService
      resource create, such as `MyEdgeService`.
    parent: Required. The parent resource of the EdgeCacheService resource.
      Must be in the format `projects/*/locations/global`.
  """
    edgeCacheService = _messages.MessageField('EdgeCacheService', 1)
    edgeCacheServiceId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)