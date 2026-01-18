from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheKeysetsGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEdgeCacheKeysetsGetRequest object.

  Fields:
    name: Required. The name of the EdgeCacheKeyset resource to get. Must be
      in the format `projects/*/locations/global/edgeCacheKeysets/*`.
  """
    name = _messages.StringField(1, required=True)