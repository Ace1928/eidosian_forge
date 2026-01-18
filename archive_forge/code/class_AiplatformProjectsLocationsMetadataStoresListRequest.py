from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresListRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresListRequest object.

  Fields:
    pageSize: The maximum number of Metadata Stores to return. The service may
      return fewer. Must be in range 1-1000, inclusive. Defaults to 100.
    pageToken: A page token, received from a previous
      MetadataService.ListMetadataStores call. Provide this to retrieve the
      subsequent page. When paginating, all other provided parameters must
      match the call that provided the page token. (Otherwise the request will
      fail with INVALID_ARGUMENT error.)
    parent: Required. The Location whose MetadataStores should be listed.
      Format: `projects/{project}/locations/{location}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)